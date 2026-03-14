"""
metrics.py  —  Aggregates per-entity verdicts into output schema
════════════════════════════════════════════════════════════════
Uses ACTUAL field names: entity, entity_type, assertion, temporality, subject
"""

ENTITY_TYPES    = ["MEDICINE","PROBLEM","PROCEDURE","TEST","VITAL_NAME",
                   "IMMUNIZATION","MEDICAL_DEVICE","MENTAL_STATUS","SDOH","SOCIAL_HISTORY"]
ASSERTION_TYPES = ["POSITIVE","NEGATIVE","UNCERTAIN"]
TEMP_TYPES      = ["CURRENT","CLINICAL_HISTORY","UPCOMING","UNCERTAIN"]
SUBJECT_TYPES   = ["PATIENT","FAMILY_MEMBER"]


class MetricsAggregator:

    def aggregate(self, file_name: str, entities: list, verdicts: list) -> dict:
        n = len(entities)
        if n == 0:
            return self._zero_report(file_name)

        type_b   = {et: [0,0] for et in ENTITY_TYPES}
        assert_b = {a:  [0,0] for a  in ASSERTION_TYPES}
        temp_b   = {t:  [0,0] for t  in TEMP_TYPES}
        subj_b   = {s:  [0,0] for s  in SUBJECT_TYPES}

        date_total = date_errors = 0
        total_fields = missing_fields = 0

        for entity, verdict in zip(entities, verdicts):
            # ACTUAL field names
            et   = str(entity.get("entity_type",  "")).upper()
            asr  = str(entity.get("assertion",    "")).upper()
            tmp  = str(entity.get("temporality",  "")).upper()
            subj = str(entity.get("subject",      "")).upper()
            qa   = entity.get("metadata_from_qa", {})

            et   = et   if et   in ENTITY_TYPES    else "PROBLEM"
            asr  = asr  if asr  in ASSERTION_TYPES else "POSITIVE"
            tmp  = tmp  if tmp  in TEMP_TYPES      else "CURRENT"
            subj = subj if subj in SUBJECT_TYPES   else "PATIENT"

            type_b[et][0]   += 1
            if verdict.get("entity_type_error"):  type_b[et][1]   += 1

            assert_b[asr][0] += 1
            if verdict.get("assertion_error"):    assert_b[asr][1] += 1

            temp_b[tmp][0]   += 1
            if verdict.get("temporality_error"):  temp_b[tmp][1]   += 1

            subj_b[subj][0]  += 1
            if verdict.get("subject_error"):      subj_b[subj][1]  += 1

            # Date accuracy: count entities that have a date in QA
            if isinstance(qa, dict):
                has_date = any(r.get("entity_type") in ("exact_date","derived_date")
                               for r in qa.get("relations", []))
                if has_date:
                    date_total += 1
                    if verdict.get("event_date_error"):
                        date_errors += 1

            # Completeness
            from rule_engine import REQUIRED_FIELDS_BY_TYPE
            req = REQUIRED_FIELDS_BY_TYPE.get(et, REQUIRED_FIELDS_BY_TYPE["PROBLEM"])
            total_fields   += len(req)
            missing_fields += len(verdict.get("missing_attributes", []))

        def rate(b):
            tot, err = b
            return round(err/tot, 4) if tot > 0 else 0.0

        return {
            "file_name":               file_name,
            "entity_type_error_rate":  {et: rate(type_b[et])   for et in ENTITY_TYPES},
            "assertion_error_rate":    {a:  rate(assert_b[a])  for a  in ASSERTION_TYPES},
            "temporality_error_rate":  {t:  rate(temp_b[t])    for t  in TEMP_TYPES},
            "subject_error_rate":      {s:  rate(subj_b[s])    for s  in SUBJECT_TYPES},
            "event_date_accuracy":     round(1.0 - (date_errors/date_total
                                              if date_total > 0 else 0.0), 4),
            "attribute_completeness":  round(1.0 - (missing_fields/total_fields
                                              if total_fields > 0 else 0.0), 4),
            "_meta": {
                "total_entities":   n,
                "entities_with_dates": date_total,
                "entity_breakdown": {et: type_b[et][0] for et in ENTITY_TYPES},
            }
        }

    def _zero_report(self, file_name):
        return {
            "file_name":               file_name,
            "entity_type_error_rate":  {et: 0.0 for et in ENTITY_TYPES},
            "assertion_error_rate":    {a:  0.0 for a  in ASSERTION_TYPES},
            "temporality_error_rate":  {t:  0.0 for t  in TEMP_TYPES},
            "subject_error_rate":      {s:  0.0 for s  in SUBJECT_TYPES},
            "event_date_accuracy":     1.0,
            "attribute_completeness":  1.0,
            "_meta": {"total_entities": 0}
        }
