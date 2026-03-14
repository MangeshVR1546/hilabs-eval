"""
context_extractor.py
════════════════════
Context extraction — now MUCH simpler because each entity already carries
its own `text` field (full surrounding sentence from the chart).

The markdown (.md) file is NOT needed — context lives inside the JSON.

This module still accepts an optional markdown string for backward
compatibility, but primarily uses entity["text"] + entity["heading"].
"""


class ContextExtractor:

    def get_context(self, entity: dict, markdown: str = "", window: int = 3) -> str:
        """
        Build context string for an entity.
        Primary source: entity['text'] (always present in JSON).
        Secondary: heading (section name).
        Tertiary: markdown (if provided, for broader window).
        """
        text_field = str(entity.get("text", "")).strip()
        heading    = str(entity.get("heading", "")).strip()

        # Clean up heading format: "Medications__page_no__1" → "Medications"
        clean_heading = heading.split("__")[0].replace("_", " ").strip()

        # Primary context: always the entity's own text field
        if text_field:
            if clean_heading:
                return f"[SECTION: {clean_heading}] {text_field}"
            return text_field

        # Fallback: search markdown for entity text
        entity_text = str(entity.get("entity", "")).strip()
        if markdown and entity_text:
            lines = markdown.split("\n")
            for i, line in enumerate(lines):
                if entity_text.lower() in line.lower():
                    start = max(0, i - window)
                    end   = min(len(lines), i + window + 1)
                    snippet = " ".join(lines[start:end]).strip()
                    if clean_heading:
                        return f"[SECTION: {clean_heading}] {snippet}"
                    return snippet

        return clean_heading or entity_text or ""
