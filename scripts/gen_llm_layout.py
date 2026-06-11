"""
LLM-direct layout baseline: ask GPT-4o-mini to predict per-shot entity bboxes
from the multi-shot narrative (the "just let an LLM make the boxes" baseline).

Output: {story_name: [ {entity: [x1,y1,x2,y2]} per shot ]}  (normalized 0-1)
Saved to outputs/layouts/llm_<tag>.json  (cached; re-run skips done stories).

Usage:
  OPENAI_API_KEY=... python scripts/gen_llm_layout.py \
    --stories data/captions/stories_aaai_eval_120.json --out outputs/layouts/llm_aaai.json
"""
import sys, os, argparse, json, re, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from openai import OpenAI

SYS = (
    "You are a layout planner for multi-shot image generation. For EACH shot you are "
    "given the entities that are PRESENT. Output a bounding box for every present entity "
    "on a normalized [0,1] canvas (origin top-left), format [x1,y1,x2,y2]. Boxes should "
    "reflect a plausible scene: distinct non-identical placements, sizes matching the "
    "object's typical scale (e.g. a bus larger than a cat), and entities that co-occur "
    "should be arranged so both are visible. Respond with JSON only."
)


def build_user(story):
    ents = {e["name"]: e["prompt"] for e in story["entities"]}
    lines = [f"Story: {story['name']}", f"Background: {story.get('background','')}",
             f"Entities: {json.dumps(ents)}", "Shots (present entities per shot):"]
    for i, sh in enumerate(story["shots"]):
        lines.append(f"  shot {i}: prompt='{sh['prompt']}' present={sh['present']}")
    lines.append('Return JSON: {"shots":[{"<entity>":[x1,y1,x2,y2], ...}, ...]} '
                 "with one object per shot (only present entities).")
    return "\n".join(lines)


def parse(content, story):
    m = re.search(r"\{.*\}", content, re.S)
    obj = json.loads(m.group(0)) if m else {}
    shots = obj.get("shots", [])
    out = []
    for i, sh in enumerate(story["shots"]):
        boxes = shots[i] if i < len(shots) and isinstance(shots[i], dict) else {}
        clean = {}
        for e in sh["present"]:
            b = boxes.get(e)
            if isinstance(b, list) and len(b) == 4:
                x1, y1, x2, y2 = [float(v) for v in b]
                x1, x2 = sorted([min(max(x1, 0), 1), min(max(x2, 0), 1)])
                y1, y2 = sorted([min(max(y1, 0), 1), min(max(y2, 0), 1)])
                if x2 - x1 < 0.03: x2 = min(1.0, x1 + 0.2)
                if y2 - y1 < 0.03: y2 = min(1.0, y1 + 0.2)
                clean[e] = [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]
            else:  # fallback center box
                clean[e] = [0.3, 0.3, 0.7, 0.9]
        out.append(clean)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stories", default="data/captions/stories_aaai_eval_120.json")
    ap.add_argument("--out", default="outputs/layouts/llm_aaai.json")
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()
    base = Path(__file__).parent.parent
    outp = base / args.out; outp.parent.mkdir(parents=True, exist_ok=True)
    stories = json.loads((base / args.stories).read_text())
    done = json.loads(outp.read_text()) if outp.exists() else {}
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    for k, st in enumerate(stories):
        if st["name"] in done:
            continue
        try:
            r = client.chat.completions.create(
                model=args.model, temperature=0.2, max_tokens=900,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": build_user(st)}])
            done[st["name"]] = parse(r.choices[0].message.content, st)
        except Exception as e:
            print(f"  ! {st['name']}: {repr(e)[:120]} -> fallback center")
            done[st["name"]] = [{e: [0.3, 0.3, 0.7, 0.9] for e in sh["present"]}
                                for sh in st["shots"]]
        if k % 10 == 0:
            outp.write_text(json.dumps(done))
            print(f"  [{k+1}/{len(stories)}] {st['name']}", flush=True)
    outp.write_text(json.dumps(done, indent=1))
    print(f"Done -> {outp} ({len(done)} stories)")


if __name__ == "__main__":
    main()
