"""
llm_judge.py
════════════
Layer 3: LLM-as-Judge

Uses Groq (free tier) as the primary LLM provider.
- Endpoint : https://api.groq.com/openai/v1/chat/completions
- Model    : llama-3.3-70b-versatile   (best reasoning, free tier)
- Fallback : llama3-8b-8192            (faster, if rate limited)

Why Groq?
  - 14,400 free requests/day, no credit card needed
  - 300+ tokens/second on LPU hardware (fastest inference available)
  - OpenAI-compatible API (easy to use with urllib, no SDK needed)
  - llama-3.3-70b-versatile rivals GPT-4o on clinical reasoning tasks

Design decisions:
  1. Few-shot examples baked into system prompt (6 clinical cases)
  2. temperature=0 for deterministic, reproducible verdicts
  3. Exponential backoff on rate-limit (429) errors
  4. Automatic fallback to smaller model on rate-limit
  5. Response caching to avoid duplicate API calls for identical entities
  6. Zero extra pip dependencies — only stdlib urllib used
"""

import json
import time
import logging
import re
import hashlib
import urllib.request
import urllib.error
from typing import Optional

log = logging.getLogger("LLM_JUDGE")

# ── Groq API config ───────────────────────────────────────────────────────────
GROQ_URL            = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_PRIMARY  = "llama-3.3-70b-versatile"   # Best: 70B, free tier
GROQ_MODEL_FALLBACK = "llama3-8b-8192"             # Faster fallback on rate limit


# ══════════════════════════════════════════════════════════════════════════════
# CLINICAL FEW-SHOT EXAMPLES  (baked into system prompt)
# ══════════════════════════════════════════════════════════════════════════════

FEW_SHOT_EXAMPLES = """
=== FEW-SHOT EXAMPLES ===

EXAMPLE 1 — Temporality error (CURRENT vs CLINICAL_HISTORY):
Entity: {"entity_text": "hypertension", "entity_type": "PROBLEM", "assertion": "POSITIVE", "temporality": "CURRENT", "subject": "PATIENT"}
Context: "Patient has a history of hypertension and diabetes."
Output: {"entity_type_error": false, "assertion_error": false, "temporality_error": true, "subject_error": false, "event_date_error": false, "missing_attributes": [], "reasoning": "The phrase 'history of' is a strong clinical history marker. Temporality should be CLINICAL_HISTORY, not CURRENT.", "confidence": 0.93}

EXAMPLE 2 — Subject error (PATIENT vs FAMILY_MEMBER):
Entity: {"entity_text": "breast cancer", "entity_type": "PROBLEM", "assertion": "POSITIVE", "temporality": "CLINICAL_HISTORY", "subject": "PATIENT"}
Context: "Family history: Mother had breast cancer at age 55."
Output: {"entity_type_error": false, "assertion_error": false, "temporality_error": false, "subject_error": true, "event_date_error": false, "missing_attributes": [], "reasoning": "Subject should be FAMILY_MEMBER because 'Mother had' explicitly attributes this condition to a family member, not the patient.", "confidence": 0.97}

EXAMPLE 3 — Entity type error (PROBLEM vs MEDICINE):
Entity: {"entity_text": "aspirin", "entity_type": "PROBLEM", "assertion": "POSITIVE", "temporality": "CURRENT", "subject": "PATIENT"}
Context: "Patient takes aspirin 81mg daily for cardiac prophylaxis."
Output: {"entity_type_error": true, "assertion_error": false, "temporality_error": false, "subject_error": false, "event_date_error": false, "missing_attributes": [], "reasoning": "Aspirin is a medication. Entity type should be MEDICINE, not PROBLEM.", "confidence": 0.98}

EXAMPLE 4 — Assertion error (POSITIVE vs NEGATIVE):
Entity: {"entity_text": "chest pain", "entity_type": "PROBLEM", "assertion": "POSITIVE", "temporality": "CURRENT", "subject": "PATIENT"}
Context: "Patient denies chest pain, shortness of breath, or palpitations."
Output: {"entity_type_error": false, "assertion_error": true, "temporality_error": false, "subject_error": false, "event_date_error": false, "missing_attributes": [], "reasoning": "The word 'denies' is a strong negation cue. Assertion should be NEGATIVE, not POSITIVE.", "confidence": 0.99}

EXAMPLE 5 — Temporality error (CURRENT vs UPCOMING):
Entity: {"entity_text": "colonoscopy", "entity_type": "PROCEDURE", "assertion": "POSITIVE", "temporality": "CURRENT", "subject": "PATIENT"}
Context: "Patient is scheduled for colonoscopy next month with gastroenterology."
Output: {"entity_type_error": false, "assertion_error": false, "temporality_error": true, "subject_error": false, "event_date_error": false, "missing_attributes": [], "reasoning": "The phrase 'scheduled for ... next month' marks this as a future event. Temporality should be UPCOMING.", "confidence": 0.95}

EXAMPLE 6 — All correct (no errors):
Entity: {"entity_text": "metformin", "entity_type": "MEDICINE", "assertion": "POSITIVE", "temporality": "CURRENT", "subject": "PATIENT"}
Context: "Current medications: Metformin 500mg twice daily for type 2 diabetes."
Output: {"entity_type_error": false, "assertion_error": false, "temporality_error": false, "subject_error": false, "event_date_error": false, "missing_attributes": [], "reasoning": "All fields are correct. Metformin is correctly classified as MEDICINE with POSITIVE/CURRENT/PATIENT attributes matching the source text.", "confidence": 0.96}

=== END EXAMPLES ===
"""

SYSTEM_PROMPT = f"""You are an expert clinical NLP evaluator specialising in medical entity extraction quality assurance.

Your task: Given an extracted clinical entity and its surrounding source text, determine whether the extraction is correct across 5 dimensions.

## Entity Fields to Evaluate

1. **entity_type** — Category of the clinical concept. Valid values:
   MEDICINE, PROBLEM, PROCEDURE, TEST, VITAL_NAME, IMMUNIZATION, MEDICAL_DEVICE, MENTAL_STATUS, SDOH, SOCIAL_HISTORY

2. **assertion** — Polarity of the assertion. Valid values:
   - POSITIVE  = condition is present, confirmed, or active
   - NEGATIVE  = condition is absent, denied, or ruled out
   - UNCERTAIN = condition is possible, suspected, or probable but not confirmed

3. **temporality** — When the condition occurred. Valid values:
   - CURRENT          = happening now, active, present
   - CLINICAL_HISTORY = occurred in the past; includes "history of", "previous", "prior"
   - UPCOMING         = planned for the future; includes "scheduled", "will be", "follow-up"
   - UNCERTAIN        = timing is unclear

4. **subject** — Who has the condition. Valid values:
   - PATIENT       = the patient themselves
   - FAMILY_MEMBER = a biological relative (mother, father, sibling, etc.)

5. **event_date** — If present, is the date plausible and correctly formatted?

## Clinical Reasoning Rules
- "history of", "h/o", "hx", "past medical history", "PMH", "prior" → CLINICAL_HISTORY
- "denies", "no", "not", "without", "absent", "negative for", "free of" → NEGATIVE
- "possible", "probable", "suspected", "rule out", "r/o", "may have", "likely" → UNCERTAIN
- "mother", "father", "family history", "FH", "maternal", "paternal" → FAMILY_MEMBER
- "scheduled", "plan to", "will undergo", "follow-up", "next visit", "ordered for" → UPCOMING
- Medications = MEDICINE (never PROBLEM)
- Diagnoses and symptoms = PROBLEM (never MEDICINE)
- Lab tests, imaging = TEST (not PROCEDURE)
- Surgical procedures, biopsies = PROCEDURE (not TEST)

{FEW_SHOT_EXAMPLES}

## Output Format
Respond ONLY with a valid JSON object — no preamble, no explanation outside JSON, no markdown code fences.

{{
  "entity_type_error": <boolean>,
  "assertion_error": <boolean>,
  "temporality_error": <boolean>,
  "subject_error": <boolean>,
  "event_date_error": <boolean>,
  "missing_attributes": [<list of missing/null field names>],
  "reasoning": "<1-2 sentence clinical reasoning explanation>",
  "confidence": <float 0.0-1.0>
}}"""


# ══════════════════════════════════════════════════════════════════════════════
# LLM JUDGE — GROQ BACKEND
# ══════════════════════════════════════════════════════════════════════════════

class LLMJudge:
    """
    LLM-as-Judge using Groq free tier.

    Features:
    - Primary model: llama-3.3-70b-versatile
    - Auto-fallback to llama3-8b-8192 on rate limit
    - Response cache (MD5 keyed) to avoid duplicate API calls
    - Exponential backoff with jitter on 429 errors
    - Zero extra dependencies (pure stdlib urllib)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key       = api_key
        self.call_count    = 0
        self.error_count   = 0
        self.cache_hits    = 0
        self._cache: dict  = {}
        self._use_fallback = False

        if not api_key:
            log.warning("No GROQ_API_KEY found. Set: export GROQ_API_KEY=your_key")
            log.warning("Get a free key at: https://console.groq.com")

    def judge(self, entity: dict, context: str) -> Optional[dict]:
        if not self.api_key:
            return None

        cache_key = self._make_cache_key(entity, context)
        if cache_key in self._cache:
            self.cache_hits += 1
            return self._cache[cache_key]

        user_message = (
            f"Entity to evaluate:\n{json.dumps(entity, indent=2)}\n\n"
            f'Source context:\n"""{context}"""'
        )

        for attempt in range(4):
            try:
                model  = GROQ_MODEL_FALLBACK if self._use_fallback else GROQ_MODEL_PRIMARY
                result = self._call_groq(user_message, model=model)
                if result:
                    self.call_count += 1
                    self._cache[cache_key] = result
                    return result

            except urllib.error.HTTPError as e:
                if e.code == 429:
                    self._use_fallback = True
                    wait = (2 ** attempt) + (attempt * 0.5)
                    log.warning("Groq rate limit — fallback model, waiting %.1fs", wait)
                    time.sleep(wait)
                elif e.code in (500, 502, 503):
                    wait = 2 ** attempt
                    log.warning("Groq server error %d — waiting %ds", e.code, wait)
                    time.sleep(wait)
                else:
                    log.error("Groq HTTP error %d: %s", e.code, e.reason)
                    break
            except Exception as e:
                wait = 2 ** attempt
                log.warning("LLM attempt %d failed: %s — retrying in %ds", attempt + 1, e, wait)
                time.sleep(wait)

        self.error_count += 1
        log.error("LLM judge failed for: %s", entity.get("entity_text", "?"))
        return None

    def _call_groq(self, user_message: str, model: str) -> Optional[dict]:
        payload = {
            "model":       model,
            "temperature": 0.0,
            "max_tokens":  600,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        }

        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            GROQ_URL,
            data    = data,
            headers = {
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method = "POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            response = json.loads(resp.read().decode("utf-8"))

        text = response["choices"][0]["message"]["content"]
        log.debug("Groq raw (%s): %s", model, text[:200])
        return self._parse_json_response(text)

    def _parse_json_response(self, text: str) -> Optional[dict]:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"```\s*$",           "", text, flags=re.MULTILINE)
        text = text.strip()

        try:
            return self._validate_and_normalise(json.loads(text))
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return self._validate_and_normalise(json.loads(match.group()))
            except json.JSONDecodeError:
                pass

        log.error("Could not parse LLM JSON: %s", text[:300])
        return None

    def _validate_and_normalise(self, result: dict) -> Optional[dict]:
        required = {
            "entity_type_error", "assertion_error",
            "temporality_error", "subject_error", "event_date_error"
        }
        for field in required:
            if field not in result:
                result[field] = False
            val = result[field]
            if isinstance(val, str):
                result[field] = val.lower() in ("true", "1", "yes")
            else:
                result[field] = bool(val)

        if "missing_attributes" not in result or not isinstance(result["missing_attributes"], list):
            result["missing_attributes"] = []

        result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.75))))
        result.setdefault("reasoning", "")
        return result

    def _make_cache_key(self, entity: dict, context: str) -> str:
        payload = json.dumps(entity, sort_keys=True) + "|" + context[:200]
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def stats(self) -> dict:
        return {
            "llm_calls":  self.call_count,
            "llm_errors": self.error_count,
            "cache_hits": self.cache_hits,
            "model":      GROQ_MODEL_FALLBACK if self._use_fallback else GROQ_MODEL_PRIMARY,
        }
