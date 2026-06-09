import json
import re


def parse_plan_output(text: str) -> dict | None:
    """Try to extract JSON from LM output."""
    text = text.strip()
    # try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # try to find JSON block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def normalize_entity_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r'^(a|an|the)\s+', '', name)
    name = name.replace(' ', '_')
    return name


def normalize_plan(plan: dict) -> dict:
    """Normalize entity names and relations."""
    entities = [normalize_entity_name(e) for e in plan.get("entities", [])]
    plan["entities"] = entities
    for shot in plan.get("shots", []):
        shot["active_entities"] = [normalize_entity_name(e) for e in shot.get("active_entities", [])]
        normalized_rels = []
        for rel in shot.get("relations", []):
            if len(rel) == 3:
                normalized_rels.append([
                    normalize_entity_name(rel[0]),
                    rel[1].lower().strip(),
                    normalize_entity_name(rel[2]),
                ])
        shot["relations"] = normalized_rels
        focus = shot.get("focus", "none")
        if focus not in ("both", "none"):
            focus = normalize_entity_name(focus)
        shot["focus"] = focus
    return plan
