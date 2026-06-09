PLAN_SYSTEM_PROMPT = """You are a strict video shot parser.
Given a sequence of shot captions, extract only visible physical entities.
Do not infer invisible entities.
Return valid JSON only.

For each shot, provide:
- background: short phrase describing the scene background
- active_entities: list of visible entity names (simple nouns, e.g. "cat", "dog")
- relations: list of [subject, relation, object] triples
- focus: entity name with main focus, or "both", or "none"

Allowed relation labels:
left_of, right_of, above, below, near, beside, overlapping, holding, riding, none

Output JSON schema exactly:
{
  "entities": [string],
  "shots": [
    {
      "shot_id": int,
      "background": string,
      "active_entities": [string],
      "relations": [[string, string, string]],
      "focus": string
    }
  ]
}"""


def build_plan_prompt(shot_captions: list[str]) -> str:
    lines = "\n".join(f"Shot {i}: {cap}" for i, cap in enumerate(shot_captions))
    return f"{PLAN_SYSTEM_PROMPT}\n\nInput:\n{lines}\n\nOutput JSON:"
