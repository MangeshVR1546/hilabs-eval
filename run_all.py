"""
run_all.py  —  Batch runner for all 30 test files
══════════════════════════════════════════════════
Usage:
  python run_all.py                        # with LLM (needs GROQ_API_KEY)
  python run_all.py --no-llm              # rule-based only, instant
  python run_all.py --data-dir /path/to/test_data
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from collections import defaultdict

from test import ClinicalEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("BATCH")


def run_all(data_dir="test_data", output_dir="output", use_llm=True):
    data_path   = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all JSON files in test_data (handles flat or nested structure)
    json_files = sorted(data_path.rglob("*.json"))
    log.info("Found %d JSON files in %s", len(json_files), data_dir)

    if not json_files:
        log.error("No JSON files found. Check --data-dir path.")
        sys.exit(1)

    evaluator = ClinicalEvaluator(use_llm=use_llm)
    results   = []
    errors    = []
    t0        = time.time()

    for json_file in json_files:
        file_name = json_file.stem
        out_file  = output_path / f"{file_name}.json"
        try:
            report = evaluator.evaluate_file(str(json_file))
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            results.append((file_name, report))
            log.info("  ✓ %s", file_name)
        except Exception as e:
            errors.append(file_name)
            log.error("  ✗ %s: %s", file_name, e)

    elapsed = time.time() - t0
    log.info("══════════════════════════════════════")
    log.info("Done: %d/%d in %.1fs", len(results), len(json_files), elapsed)
    if errors:
        log.warning("Failed: %s", errors)

    if results:
        _print_summary(results)
        _save_summary(results, output_path)


def _print_summary(results):
    from metrics import ENTITY_TYPES, ASSERTION_TYPES, TEMP_TYPES, SUBJECT_TYPES
    agg = defaultdict(list)
    for _, r in results:
        for et, v in r.get("entity_type_error_rate",  {}).items(): agg[f"type_{et}"].append(v)
        for a,  v in r.get("assertion_error_rate",    {}).items(): agg[f"asr_{a}"].append(v)
        for t,  v in r.get("temporality_error_rate",  {}).items(): agg[f"tmp_{t}"].append(v)
        for s,  v in r.get("subject_error_rate",      {}).items(): agg[f"subj_{s}"].append(v)
        agg["date_acc"].append(r.get("event_date_accuracy",    1.0))
        agg["attr_comp"].append(r.get("attribute_completeness",1.0))

    def avg(k): lst=agg[k]; return round(sum(lst)/len(lst),4) if lst else 0.0

    print("\n" + "="*55)
    print("  AGGREGATE SUMMARY (mean across all files)")
    print("="*55)
    print("\n📊 Entity Type Error Rates:")
    for et in ENTITY_TYPES:
        v = avg(f"type_{et}")
        bar = "█" * int(v * 20)
        print(f"   {et:<20} {v:>6.2%}  {bar}")
    print("\n📊 Assertion Error Rates:")
    for a in ASSERTION_TYPES:
        print(f"   {a:<20} {avg(f'asr_{a}'):>6.2%}")
    print("\n📊 Temporality Error Rates:")
    for t in TEMP_TYPES:
        print(f"   {t:<20} {avg(f'tmp_{t}'):>6.2%}")
    print("\n📊 Subject Error Rates:")
    for s in SUBJECT_TYPES:
        print(f"   {s:<20} {avg(f'subj_{s}'):>6.2%}")
    print(f"\n📊 Event Date Accuracy:    {avg('date_acc'):>6.2%}")
    print(f"📊 Attribute Completeness: {avg('attr_comp'):>6.2%}")
    print("="*55 + "\n")


def _save_summary(results, output_path):
    """Save a machine-readable summary JSON for use in report generation."""
    from metrics import ENTITY_TYPES, ASSERTION_TYPES, TEMP_TYPES, SUBJECT_TYPES
    agg = defaultdict(list)
    for _, r in results:
        for et, v in r.get("entity_type_error_rate",  {}).items(): agg[f"type_{et}"].append(v)
        for a,  v in r.get("assertion_error_rate",    {}).items(): agg[f"asr_{a}"].append(v)
        for t,  v in r.get("temporality_error_rate",  {}).items(): agg[f"tmp_{t}"].append(v)
        for s,  v in r.get("subject_error_rate",      {}).items(): agg[f"subj_{s}"].append(v)
        agg["date_acc"].append(r.get("event_date_accuracy",    1.0))
        agg["attr_comp"].append(r.get("attribute_completeness",1.0))

    def avg(k): lst=agg[k]; return round(sum(lst)/len(lst),4) if lst else 0.0

    summary = {
        "files_evaluated": len(results),
        "entity_type_error_rate":  {et: avg(f"type_{et}") for et in ENTITY_TYPES},
        "assertion_error_rate":    {a:  avg(f"asr_{a}")  for a  in ASSERTION_TYPES},
        "temporality_error_rate":  {t:  avg(f"tmp_{t}")  for t  in TEMP_TYPES},
        "subject_error_rate":      {s:  avg(f"subj_{s}") for s  in SUBJECT_TYPES},
        "event_date_accuracy":     avg("date_acc"),
        "attribute_completeness":  avg("attr_comp"),
    }
    summary_path = output_path / "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary saved → %s", summary_path)


if __name__ == "__main__":
    use_llm  = "--no-llm" not in sys.argv
    data_dir = "test_data"
    out_dir  = "output"
    for i, arg in enumerate(sys.argv):
        if arg == "--data-dir"   and i+1 < len(sys.argv): data_dir = sys.argv[i+1]
        if arg == "--output-dir" and i+1 < len(sys.argv): out_dir  = sys.argv[i+1]
    run_all(data_dir=data_dir, output_dir=out_dir, use_llm=use_llm)
