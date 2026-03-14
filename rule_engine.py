"""
rule_engine.py  —  Layer 1: Rule-Based Evaluation
══════════════════════════════════════════════════
Uses ACTUAL field names from the Entity JSON schema:
  entity          → extracted entity text
  entity_type     → MEDICINE, PROBLEM, PROCEDURE, TEST, VITAL_NAME, ...
  assertion       → POSITIVE, NEGATIVE, UNCERTAIN, ""
  temporality     → CURRENT, CLINICAL_HISTORY, UPCOMING, UNCERTAIN, ""
  subject         → PATIENT, FAMILY_MEMBER, ""
  heading         → section header (e.g. "Medications__page_no__1")
  text            → full surrounding context sentence(s)
  metadata_from_qa→ {relations:[...], count:int}  or  {}
"""

import re
from typing import Optional

# ── Negation lexicons ─────────────────────────────────────────────────────────
NEGATION_PRE = [
    r"\bno\b", r"\bnot\b", r"\bdenies?\b", r"\bdeny\b", r"\bwithout\b",
    r"\babsent\b", r"\bnegative\s+for\b", r"\bfree\s+of\b", r"\bnever\b",
    r"\bruled?\s+out\b", r"\bno\s+evidence\s+of\b", r"\bno\s+sign\s+of\b",
    r"\bdoes\s+not\s+have\b", r"\bdid\s+not\s+have\b", r"\bnon[-\s]?\b",
    r"\bwithhold\b", r"\bwithdrawn\b", r"\bdeclines?\b", r"\brefuses?\b",
]
NEGATION_POST = [
    r"\bunlikely\b", r"\bexcluded\b", r"\bwas\s+ruled\s+out\b",
    r"\bnot\s+found\b", r"\bnot\s+detected\b", r"\bnot\s+present\b",
    r"\bnot\s+seen\b", r"\bnot\s+identified\b",
]
UNCERTAINTY_TRIGGERS = [
    r"\bpossible\b", r"\bpossibly\b", r"\bprobable\b", r"\bprobably\b",
    r"\bsuspect\b", r"\bsuspected\b", r"\bsuspicion\s+of\b", r"\blikely\b",
    r"\bmay\s+have\b", r"\bmight\b", r"\bcould\s+be\b", r"\bquestionable\b",
    r"\bunclear\b", r"\bworrisome\s+for\b", r"\bconcerning\s+for\b",
    r"\brule\s+out\b", r"\br/o\b", r"\bapparent\b", r"\bpresumably\b",
]
HISTORICAL_TRIGGERS = [
    r"\bhistory\s+of\b", r"\bhx\s+of\b", r"\bh/o\b", r"\bpast\s+history\b",
    r"\bmedical\s+history\b", r"\bpmh\b", r"\bprevious\b", r"\bpreviously\b",
    r"\bformer\b", r"\bformerly\b", r"\bprior\b", r"\bwas\s+diagnosed\b",
    r"\bchronic\b", r"\bknown\s+history\b", r"\byears?\s+ago\b",
    r"\bmonths?\s+ago\b", r"\bweeks?\s+ago\b", r"\bremote\s+history\b",
    r"\bpast\s+medical\b",
]
UPCOMING_TRIGGERS = [
    r"\bscheduled\b", r"\bwill\s+be\b", r"\bplanned\b", r"\bplan\s+to\b",
    r"\bfollow[\s-]up\b", r"\bfuture\b", r"\bupcoming\b", r"\bpending\b",
    r"\bto\s+be\s+done\b", r"\bwill\s+undergo\b", r"\bnext\s+visit\b",
    r"\border(?:ed)?\s+for\b", r"\breferral\s+for\b",
]
FAMILY_TRIGGERS = [
    r"\bmother\b", r"\bfather\b", r"\bbrother\b", r"\bsister\b",
    r"\bgrandmother\b", r"\bgrandfather\b", r"\baunt\b", r"\buncle\b",
    r"\bcousin\b", r"\bparent\b", r"\bparents\b", r"\bsibling\b",
    r"\bson\b", r"\bdaughter\b", r"\bchildren\b",
    r"\bfamily\s+history\b", r"\bfh\b", r"\bfhx\b",
    r"\bmaternal\b", r"\bpaternal\b", r"\brelative\b",
    r"\bspouse\b", r"\bhusband\b", r"\bwife\b",
]

# ── Heading → expected temporality ────────────────────────────────────────────
HEADING_TEMPORALITY = {
    r"past.medical|pmh|surgical.history|psh|medical.history": "CLINICAL_HISTORY",
    r"family.history|fhx|fh\b":                               "CLINICAL_HISTORY",
    r"current.med|medication|active.med|allerg":              "CURRENT",
    r"chief.complaint|present.illness|hpi":                   "CURRENT",
    r"vital|physical.exam|review.of.system|ros\b":            "CURRENT",
    r"plan|follow.?up|discharge.instruction":                 "UPCOMING",
    r"social.history":                                        "CURRENT",
}
# ── Heading → expected subject ────────────────────────────────────────────────
HEADING_SUBJECT = {
    r"family.history|fhx|fh\b": "FAMILY_MEMBER",
}

# ── Vocabulary validators per entity type ─────────────────────────────────────
ENTITY_TYPE_VALIDATORS = {
    "MEDICINE": [
        r"\bmg\b", r"\bml\b", r"\btablet\b", r"\bcapsule\b", r"\bdose\b",
        r"\bdaily\b", r"\bbid\b", r"\btid\b", r"\bqid\b", r"\bprn\b",
        r"\bpo\b", r"\biv\b", r"\bpatch\b", r"\binjection\b", r"\borally\b",
    ],
    "TEST": [
        r"\btest\b", r"\blevel\b", r"\bcount\b", r"\bpanel\b", r"\bculture\b",
        r"\bscan\b", r"\bx[\s-]?ray\b", r"\bmri\b", r"\bct\b",
        r"\bultrasound\b", r"\becg\b", r"\blab\b", r"\bblood\b",
        r"\bglucose\b", r"\bhemoglobin\b", r"\bcreatinine\b",
    ],
    "VITAL_NAME": [
        r"\bpressure\b", r"\bpulse\b", r"\btemperature\b", r"\bweight\b",
        r"\bheight\b", r"\bbmi\b", r"\bsaturation\b", r"\bspo2\b",
        r"\brespiratory\b", r"\brate\b", r"\bbp\b", r"\bhr\b",
        r"\bmmhg\b", r"\bbpm\b",
    ],
}

# ── Required fields per entity type (ACTUAL field names) ──────────────────────
REQUIRED_FIELDS_BY_TYPE = {
    "MEDICINE":       ["entity", "assertion", "temporality", "subject", "text"],
    "PROBLEM":        ["entity", "assertion", "temporality", "subject", "text"],
    "PROCEDURE":      ["entity", "assertion", "temporality", "subject", "text"],
    "TEST":           ["entity", "assertion", "temporality", "subject", "text"],
    "VITAL_NAME":     ["entity", "assertion", "temporality", "text"],
    "IMMUNIZATION":   ["entity", "assertion", "temporality", "subject", "text"],
    "MEDICAL_DEVICE": ["entity", "assertion", "temporality", "subject", "text"],
    "MENTAL_STATUS":  ["entity", "assertion", "temporality", "subject", "text"],
    "SDOH":           ["entity", "assertion", "temporality", "subject", "text"],
    "SOCIAL_HISTORY": ["entity", "assertion", "temporality", "subject", "text"],
}

# ── Expected QA sub-fields per entity type ────────────────────────────────────
EXPECTED_QA_FIELDS = {
    "MEDICINE":   {"STRENGTH", "UNIT", "FREQUENCY"},
    "TEST":       {"TEST_VALUE", "TEST_UNIT"},
    "VITAL_NAME": {"VITAL_NAME_VALUE", "VITAL_NAME_UNIT"},
}

VALID_ENTITY_TYPES = set(REQUIRED_FIELDS_BY_TYPE.keys())
VALID_ASSERTIONS   = {"POSITIVE", "NEGATIVE", "UNCERTAIN"}
VALID_TEMPORALITY  = {"CURRENT", "CLINICAL_HISTORY", "UPCOMING", "UNCERTAIN"}
VALID_SUBJECTS     = {"PATIENT", "FAMILY_MEMBER"}


class RuleEngine:

    def evaluate(self, entity: dict, context: str) -> dict:
        # Use ACTUAL field names
        entity_text  = str(entity.get("entity", "")).lower()
        entity_type  = str(entity.get("entity_type", "")).upper()
        assertion    = str(entity.get("assertion", "")).upper()
        temporality  = str(entity.get("temporality", "")).upper()
        subject      = str(entity.get("subject", "")).upper()
        heading      = str(entity.get("heading", "")).lower()
        text_field   = str(entity.get("text", "")).lower()
        qa_data      = entity.get("metadata_from_qa", {})

        # Combined context = entity's own text field + any extra context passed in
        combined_ctx = f"{text_field} {context}".lower()

        et_error, et_conf = self._check_entity_type(entity_type, entity_text, combined_ctx)
        a_error,  a_conf  = self._check_assertion(assertion, combined_ctx)
        t_error,  t_conf  = self._check_temporality(temporality, combined_ctx, heading)
        s_error,  s_conf  = self._check_subject(subject, combined_ctx, heading)
        d_error           = self._check_event_date_from_qa(qa_data)
        missing           = self._check_completeness(entity, entity_type, qa_data)

        confidence = 0.30*et_conf + 0.25*a_conf + 0.25*t_conf + 0.20*s_conf

        return {
            "entity_type_error":  et_error,
            "assertion_error":    a_error,
            "temporality_error":  t_error,
            "subject_error":      s_error,
            "event_date_error":   d_error,
            "missing_attributes": missing,
            "confidence":         confidence,
            "entity_type":        entity_type or "UNKNOWN",
            "assertion":          assertion   or "",
            "temporality":        temporality or "",
            "subject":            subject     or "",
        }

    def _check_entity_type(self, et, entity_text, ctx):
        if not et or et not in VALID_ENTITY_TYPES:
            return True, 0.9
        if et not in ENTITY_TYPE_VALIDATORS:
            return False, 0.5
        combined = f"{entity_text} {ctx}"
        hits = sum(1 for p in ENTITY_TYPE_VALIDATORS[et]
                   if re.search(p, combined, re.IGNORECASE))
        if hits >= 2: return False, 0.85
        if hits == 1: return False, 0.60
        return True, 0.70

    def _check_assertion(self, assertion, ctx):
        if assertion == "":
            return False, 0.5
        if assertion not in VALID_ASSERTIONS:
            return True, 0.95
        neg_pre  = sum(1 for p in NEGATION_PRE  if re.search(p, ctx, re.IGNORECASE))
        neg_post = sum(1 for p in NEGATION_POST if re.search(p, ctx, re.IGNORECASE))
        uncert   = sum(1 for p in UNCERTAINTY_TRIGGERS if re.search(p, ctx, re.IGNORECASE))
        has_neg, has_unc = (neg_pre+neg_post) > 0, uncert > 0

        if assertion == "POSITIVE" and (neg_pre+neg_post) >= 2: return True, 0.85
        if assertion == "POSITIVE" and has_neg and not has_unc: return True, 0.70
        if assertion == "NEGATIVE" and not has_neg:             return True, 0.60
        if assertion == "UNCERTAIN" and not has_unc and not has_neg: return True, 0.55
        return False, 0.75

    def _check_temporality(self, temporality, ctx, heading):
        if temporality == "":
            return False, 0.5
        if temporality not in VALID_TEMPORALITY:
            return True, 0.95
        heading_hint = self._heading_temporality(heading)
        if heading_hint and heading_hint != temporality:
            return True, 0.80
        hist     = sum(1 for p in HISTORICAL_TRIGGERS if re.search(p, ctx, re.IGNORECASE))
        upcoming = sum(1 for p in UPCOMING_TRIGGERS   if re.search(p, ctx, re.IGNORECASE))
        if temporality == "CURRENT" and hist >= 2:             return True, 0.80
        if temporality == "CURRENT" and hist==1 and upcoming==0: return True, 0.60
        if temporality == "CURRENT" and upcoming >= 2:         return True, 0.80
        if temporality == "CLINICAL_HISTORY" and hist==0 and upcoming==0: return True, 0.55
        if temporality == "UPCOMING" and upcoming == 0:        return True, 0.60
        return False, 0.75

    def _check_subject(self, subject, ctx, heading):
        if subject == "":
            return False, 0.5
        if subject not in VALID_SUBJECTS:
            return True, 0.95
        heading_hint = self._heading_subject(heading)
        if heading_hint == "FAMILY_MEMBER" and subject == "PATIENT":
            return True, 0.85
        family_hits = sum(1 for p in FAMILY_TRIGGERS if re.search(p, ctx, re.IGNORECASE))
        if subject == "PATIENT" and family_hits >= 2: return True, 0.85
        if subject == "PATIENT" and family_hits == 1: return True, 0.60
        if subject == "FAMILY_MEMBER" and family_hits == 0: return True, 0.65
        return False, 0.80

    def _check_event_date_from_qa(self, qa_data):
        if not qa_data or not isinstance(qa_data, dict):
            return False
        for rel in qa_data.get("relations", []):
            if rel.get("entity_type") in ("exact_date", "derived_date"):
                d = str(rel.get("entity", "")).strip()
                if re.match(r"^\d{4}-\d{2}-\d{2}$", d):
                    y, m, day = int(d[:4]), int(d[5:7]), int(d[8:])
                    if not (1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= day <= 31):
                        return True
                elif d and not any(re.search(p, d, re.IGNORECASE) for p in [
                    r"\d{4}-\d{2}-\d{2}", r"\d{1,2}/\d{1,2}/\d{2,4}",
                    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b"
                ]):
                    return True
        return False

    def _check_completeness(self, entity, entity_type, qa_data):
        et       = (entity_type or "PROBLEM").upper()
        required = REQUIRED_FIELDS_BY_TYPE.get(et, REQUIRED_FIELDS_BY_TYPE["PROBLEM"])
        missing  = [f for f in required
                    if entity.get(f) in (None, "", [], {})]
        # QA sub-field check
        expected_qa = EXPECTED_QA_FIELDS.get(et, set())
        if expected_qa and isinstance(qa_data, dict):
            found = {r.get("entity_type") for r in qa_data.get("relations", [])}
            for qf in expected_qa:
                if qf not in found:
                    missing.append(f"qa:{qf}")
        return missing

    def _heading_temporality(self, heading) -> Optional[str]:
        for pat, hint in HEADING_TEMPORALITY.items():
            if re.search(pat, heading, re.IGNORECASE):
                return hint
        return None

    def _heading_subject(self, heading) -> Optional[str]:
        for pat, hint in HEADING_SUBJECT.items():
            if re.search(pat, heading, re.IGNORECASE):
                return hint
        return None
