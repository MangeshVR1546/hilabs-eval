"""
╔══════════════════════════════════════════════════════════════════════╗
║       CLINICAL AI EVALUATION FRAMEWORK — HiLabs Workshop            ║
║       Triple-Layer Hybrid Evaluation Engine                         ║
║                                                                      ║
║  Layer 1: Rule-Based (NegEx-style)  — fast, deterministic           ║
║  Layer 2: Heuristic Signal Scoring  — pattern + QA-aware            ║
║  Layer 3: LLM-as-Judge (Groq)       — deep reasoning, hard cases    ║
║                                                                      ║
║  Usage:                                                              ║
║    python test.py <input.json> <output.json>                         ║
║    python test.py <input.json> <output.json> --no-llm               ║
╚══════════════════════════════════════════════════════════════════════╝

JSON Schema used (actual field names):
  entity, entity_type, assertion, temporality, subject,
  heading, text, metadata_from_qa
"""

import sys
import json
import os
import time
import logging
from pathlib import Path
from typing import Optional

from rule_engine       import RuleEngine
from heuristic_engine  import HeuristicEngine
from llm_judge         import LLMJudge
from metrics           import MetricsAggregator
from context_extractor import ContextExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("EVAL")


class ClinicalEvaluator:
    """
    Triple-Layer Hybrid Evaluator.

    Confidence gate:
      - Rule engine returns a confidence score 0-1
      - If confidence < threshold → escalate to LLM judge
      - Heuristic signals always run as soft modifiers
    """

    def __init__(self, use_llm: bool = True, llm_threshold: float = 0.55):
        self.rule_engine  = RuleEngine()
        self.heuristic    = HeuristicEngine()
        self.extractor    = ContextExtractor()
        self.metrics      = MetricsAggregator()
        self.use_llm      = use_llm
        self.llm_threshold = llm_threshold

        if use_llm:
            api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("LLM_API_KEY")
            self.llm = LLMJudge(api_key=api_key)
        else:
            self.llm = None

        log.info("ClinicalEvaluator ready | LLM=%s | threshold=%.2f",
                 "ON" if use_llm else "OFF", llm_threshold)

    # ── Public ────────────────────────────────────────────────────────────────

    def evaluate_file(self, input_json_path: str) -> dict:
        input_path = Path(input_json_path)
        file_name  = input_path.stem

        log.info("▶ Evaluating: %s", file_name)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entities = self._extract_entities(data)
        log.info("   Entities found: %d", len(entities))

        if not entities:
            log.warning("   No entities — returning zero-error report")
            return self._empty_report(file_name)

        verdicts    = []
        llm_calls   = 0

        for entity in entities:
            # Each entity already has its own 'text' field — use it as context
            context = self.extractor.get_context(entity)

            # Layer 1: Rule-based
            rule_verdict = self.rule_engine.evaluate(entity, context)

            # Layer 2: Heuristic
            heuristic    = self.heuristic.score(entity, context)

            # Merge confidence
            merged_conf  = self._merge_confidence(rule_verdict, heuristic)

            # Layer 3: LLM escalation for uncertain cases
            llm_verdict  = None
            if self.use_llm and self.llm and merged_conf < self.llm_threshold:
                llm_verdict = self.llm.judge(entity, context)
                llm_calls  += 1

            final = self._fuse(rule_verdict, heuristic, llm_verdict)
            verdicts.append(final)

        log.info("   LLM escalations: %d / %d", llm_calls, len(entities))
        report = self.metrics.aggregate(file_name, entities, verdicts)
        return report

    # ── Private ───────────────────────────────────────────────────────────────

    def _extract_entities(self, data) -> list:
        """Handle various JSON root structures."""
        if isinstance(data, list):
            return data
        for key in ("entities", "results", "extracted_entities", "data", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
        # Fallback: any list value whose items are dicts with 'entity' key
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                if "entity" in v[0] or "entity_type" in v[0]:
                    return v
        return []

    def _merge_confidence(self, rule_verdict: dict, heuristic: dict) -> float:
        rc = rule_verdict.get("confidence", 0.5)
        hc = heuristic.get("overall_signal_strength", 0.5)
        return 0.60 * rc + 0.40 * hc

    def _fuse(self, rule: dict, heuristic: dict, llm: Optional[dict]) -> dict:
        fused = {
            "entity_type_error":  rule.get("entity_type_error",  False),
            "assertion_error":    rule.get("assertion_error",    False),
            "temporality_error":  rule.get("temporality_error",  False),
            "subject_error":      rule.get("subject_error",      False),
            "event_date_error":   rule.get("event_date_error",   False),
            "missing_attributes": rule.get("missing_attributes", []),
            "entity_type":        rule.get("entity_type",  "UNKNOWN"),
            "assertion":          rule.get("assertion",    ""),
            "temporality":        rule.get("temporality",  ""),
            "subject":            rule.get("subject",      ""),
            "source":             "rule",
        }

        # LLM overrides on high-confidence verdicts
        if llm and llm.get("confidence", 0) > 0.65:
            for f in ("entity_type_error","assertion_error",
                      "temporality_error","subject_error","event_date_error"):
                if f in llm:
                    fused[f] = llm[f]
            if "missing_attributes" in llm:
                fused["missing_attributes"] = list(set(
                    fused["missing_attributes"] + llm["missing_attributes"]
                ))
            fused["source"] = "llm_override"

        # Heuristic soft corrections
        if heuristic.get("strong_negation_cue") and not fused["assertion_error"]:
            if fused["assertion"] == "POSITIVE":
                fused["assertion_error"] = True
                fused["source"] += "+heuristic_neg"

        if heuristic.get("family_member_cue") and not fused["subject_error"]:
            if fused["subject"] == "PATIENT":
                fused["subject_error"] = True
                fused["source"] += "+heuristic_family"

        if heuristic.get("historical_cue") and not fused["temporality_error"]:
            if fused["temporality"] == "CURRENT":
                fused["temporality_error"] = True
                fused["source"] += "+heuristic_temp"

        return fused

    def _empty_report(self, file_name):
        from metrics import ENTITY_TYPES, ASSERTION_TYPES, TEMP_TYPES, SUBJECT_TYPES
        return {
            "file_name":               file_name,
            "entity_type_error_rate":  {et: 0.0 for et in ENTITY_TYPES},
            "assertion_error_rate":    {a:  0.0 for a  in ASSERTION_TYPES},
            "temporality_error_rate":  {t:  0.0 for t  in TEMP_TYPES},
            "subject_error_rate":      {s:  0.0 for s  in SUBJECT_TYPES},
            "event_date_accuracy":     1.0,
            "attribute_completeness":  1.0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("Usage: python test.py <input.json> <output.json> [--no-llm]")
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2]
    use_llm     = "--no-llm" not in sys.argv

    if not Path(input_path).exists():
        log.error("Input not found: %s", input_path)
        sys.exit(1)

    evaluator = ClinicalEvaluator(use_llm=use_llm)

    t0     = time.time()
    report = evaluator.evaluate_file(input_path)
    elapsed = time.time() - t0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log.info("✓ Done in %.2fs → %s", elapsed, output_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
