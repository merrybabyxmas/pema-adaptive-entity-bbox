"""
Occlusion correctness via VLM (GPT-4o-mini vision) for occlusion/depth prompts (group D).

D-group shot prompts encode the relation explicitly, e.g.
  "a black dog in front of a red car in a park"   -> front = dog
  "a red car behind a black dog in a park"         -> front = dog (car behind)
We parse the prescribed front/back pair, ask the VLM a binary question on the generated
keyframe, and compare. "beside" shots are skipped (no occlusion order).

  VLM_Occlusion_Accuracy = correct / total
  Occlusion_Failure_Rate = 1 - accuracy
Raw responses saved to outputs/eval_120/vlm/.

Usage:
  OPENAI_API_KEY=... python scripts/eval_occlusion_vlm.py \
    --jobs FINAL_combo,B_template,B_llm --root outputs/lisa/aaai_ablation \
    --out outputs/eval_120/metrics/occlusion.json
"""
import sys, os, argparse, json, base64, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from openai import OpenAI


def parse_relation(prompt, ents):
    """return (front_name, front_prompt, back_name, back_prompt) or None."""
    p = prompt.lower()
    e = {x["name"]: x["prompt"] for x in ents}
    names = list(e)
    def pos(n):  # earliest position of entity name or its head noun
        return min([p.find(n.split("/")[0])] + [p.find(w) for w in e[n].lower().split() if len(w) > 2 and w in p] + [10**9])
    order = sorted(names, key=pos)
    if "in front of" in p:
        f, b = order[0], order[1]
    elif "behind" in p:
        f, b = order[1], order[0]  # first-mentioned is behind
    else:
        return None
    return f, e[f], b, e[b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--root", default="outputs/lisa/aaai_ablation")
    ap.add_argument("--stories", default="data/captions/stories_aaai_eval_120.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--group", default="D")
    args = ap.parse_args()
    base = Path(__file__).parent.parent
    stories = [s for s in json.loads((base / args.stories).read_text())
               if s["name"].split("_")[1][0] == args.group]
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    vlm_dir = base / "outputs/eval_120/vlm"; vlm_dir.mkdir(parents=True, exist_ok=True)

    def ask(img_path, fp, bp):
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        q = (f"Look at the image. Is the {fp} clearly in front of (occluding / closer than) "
             f"the {bp}? Answer with a single word: yes or no.")
        r = client.chat.completions.create(model=args.model, temperature=0, max_tokens=4,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": q},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}])
        return r.choices[0].message.content.strip().lower()

    out = {}
    for job in args.jobs.split(","):
        jdir = base / args.root / job
        raw, correct, total = [], 0, 0
        for st in stories:
            for t, sh in enumerate(st["shots"]):
                rel = parse_relation(sh["prompt"], st["entities"])
                if rel is None:
                    continue
                p = jdir / st["name"] / f"shot_{t:03d}.png"
                if not p.exists():
                    continue
                f, fp, b, bp = rel
                try:
                    ans = ask(p, fp, bp)
                except Exception as e:
                    ans = f"err:{repr(e)[:40]}"
                ok = ans.startswith("yes")
                correct += int(ok); total += 1
                raw.append({"story": st["name"], "shot": t, "front": f, "back": b,
                            "prompt": sh["prompt"], "answer": ans, "correct": ok})
        out[job] = {"vlm_occlusion_acc": round(correct / total, 4) if total else None,
                    "occlusion_fail": round(1 - correct / total, 4) if total else None, "_n": total}
        json.dump(raw, open(vlm_dir / f"occlusion_{job}.json", "w"), indent=1)
        print(f"  {job}: acc={out[job]['vlm_occlusion_acc']} (n={total})", flush=True)
    json.dump(out, open(base / args.out, "w"), indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
