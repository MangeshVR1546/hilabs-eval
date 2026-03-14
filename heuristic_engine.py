"""
heuristic_engine.py  —  Layer 2: Heuristic Signal Engine
══════════════════════════════════════════════════════════
Uses ACTUAL field names: entity, entity_type, assertion, temporality,
subject, heading, text, metadata_from_qa
"""

import re
import math
from typing import Optional

from rule_engine import (NEGATION_PRE, NEGATION_POST, FAMILY_TRIGGERS,
                         HISTORICAL_TRIGGERS, UPCOMING_TRIGGERS)

VITAL_RANGES = {
    r"heart.rate|hr\b|pulse":    (30, 250),
    r"blood.pressure|bp\b":      (50, 300),
    r"temperature|temp\b":       (90, 110),
    r"spo2|oxygen.sat":          (50, 100),
    r"respiratory.rate|rr\b":    (5,  60),
    r"weight\b":                 (1,  700),
    r"bmi\b":                    (5,  90),
}

CONTRADICTION_RULES = {
    ("NEGATIVE", "UPCOMING"),
    ("UNCERTAIN", "UPCOMING"),
}

OCR_NOISE_PATTERNS = [
    r"[^\x00-\x7F]",
    r"\b[A-Z]{6,}\b",
    r"[|\\\/]{2,}",
    r"\.{4,}",
    r"_{3,}",
]


class HeuristicEngine:

    def score(self, entity: dict, context: str) -> dict:
        # ACTUAL field names
        entity_text = str(entity.get("entity", "")).lower()
        assertion   = str(entity.get("assertion", "")).upper()
        temporality = str(entity.get("temporality", "")).upper()
        subject     = str(entity.get("subject", "")).upper()
        entity_type = str(entity.get("entity_type", "")).upper()
        heading     = str(entity.get("heading", "")).lower()
        text_field  = str(entity.get("text", "")).lower()
        qa_data     = entity.get("metadata_from_qa", {})

        combined_ctx = f"{text_field} {context}".lower()

        neg_score    = self._negation_proximity(entity_text, combined_ctx)
        family_cue   = self._family_cue(combined_ctx)
        hist_cue     = any(re.search(p, combined_ctx, re.IGNORECASE) for p in HISTORICAL_TRIGGERS)
        upcoming_cue = any(re.search(p, combined_ctx, re.IGNORECASE) for p in UPCOMING_TRIGGERS)
        contradiction = (assertion, temporality) in CONTRADICTION_RULES
        ocr_noise    = self._ocr_noise(entity_text + " " + combined_ctx)
        vital_err    = self._vital_range(entity_type, entity_text, combined_ctx)

        # QA-specific signals
        qa_score_issue = self._qa_low_confidence(qa_data)
        qa_value_error = self._qa_value_sanity(qa_data, entity_type)

        strength = 1.0
        if neg_score > 0.7  and assertion == "POSITIVE":  strength -= 0.30
        if family_cue       and subject == "PATIENT":      strength -= 0.25
        if hist_cue         and temporality == "CURRENT":  strength -= 0.20
        if upcoming_cue     and temporality == "CURRENT":  strength -= 0.15
        if contradiction:                                   strength -= 0.20
        if ocr_noise > 0.4:                                strength -= 0.10
        if vital_err:                                       strength -= 0.25
        if qa_score_issue:                                  strength -= 0.10
        if qa_value_error:                                  strength -= 0.15

        strength = max(0.0, min(1.0, strength))

        return {
            "overall_signal_strength": strength,
            "strong_negation_cue":     neg_score > 0.6,
            "family_member_cue":       family_cue,
            "historical_cue":          hist_cue,
            "upcoming_cue":            upcoming_cue,
            "contradiction_found":     contradiction,
            "ocr_noise_score":         ocr_noise,
            "vital_range_error":       vital_err,
            "qa_low_confidence":       qa_score_issue,
            "qa_value_error":          qa_value_error,
        }

    def _negation_proximity(self, entity_text, ctx):
        if not entity_text or not ctx:
            return 0.0
        try:
            pos = ctx.index(entity_text)
        except ValueError:
            pos = len(ctx) // 2

        pre_window  = ctx[max(0, pos-60):pos]
        post_window = ctx[pos+len(entity_text):pos+len(entity_text)+35]

        pre_score  = sum(self._dist_weight(pre_window,  p, False) for p in NEGATION_PRE)
        post_score = sum(self._dist_weight(post_window, p, True)  for p in NEGATION_POST)
        raw = pre_score + post_score * 0.5
        return 1 / (1 + math.exp(-raw + 1.5))

    def _dist_weight(self, window, pattern, forward):
        m = re.search(pattern, window, re.IGNORECASE)
        if not m:
            return 0.0
        dist = m.start() if forward else len(window) - m.end()
        return math.exp(-dist / 15.0)

    def _family_cue(self, ctx):
        return any(re.search(p, ctx, re.IGNORECASE) for p in FAMILY_TRIGGERS)

    def _ocr_noise(self, text):
        if not text:
            return 0.0
        return sum(1 for p in OCR_NOISE_PATTERNS
                   if re.search(p, text)) / len(OCR_NOISE_PATTERNS)

    def _vital_range(self, entity_type, entity_text, ctx):
        if entity_type != "VITAL_NAME":
            return False
        combined = f"{entity_text} {ctx}"
        numbers  = re.findall(r"\b(\d+(?:\.\d+)?)\b", combined)
        for pattern, (lo, hi) in VITAL_RANGES.items():
            if re.search(pattern, combined, re.IGNORECASE):
                for n in numbers:
                    if float(n) < lo or float(n) > hi:
                        return True
        return False

    def _qa_low_confidence(self, qa_data):
        """Flag if any QA relation has suspiciously low confidence score."""
        if not qa_data or not isinstance(qa_data, dict):
            return False
        for rel in qa_data.get("relations", []):
            score = rel.get("entity_score", 1.0)
            if isinstance(score, (int, float)) and score < 0.5:
                return True
        return False

    def _qa_value_sanity(self, qa_data, entity_type):
        """Check QA values make clinical sense (e.g. strength=0 for medicine)."""
        if not qa_data or entity_type not in ("MEDICINE", "TEST", "VITAL_NAME"):
            return False
        for rel in qa_data.get("relations", []):
            et  = rel.get("entity_type", "")
            val = str(rel.get("entity", "")).strip()
            if et == "STRENGTH" and val in ("0", "0.0", ""):
                return True
            if et in ("TEST_VALUE", "VITAL_NAME_VALUE"):
                try:
                    if float(val) < 0:
                        return True
                except (ValueError, TypeError):
                    pass
        return False
