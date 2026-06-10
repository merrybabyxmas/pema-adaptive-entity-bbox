"""
Planner -> Image-LISA end-to-end.

Input: a multi-shot story {shots:[text], entity_references:{name: path|null},
optional entity_prompts:{name: desc}}.

Planner stage (no hand-authored layout):
  - PRESENCE: for each shot, which entities appear  -> keyword match of entity
    names in the shot text (rule-based; the neural bbox-planner ckpt was lost to
    cleanup, this fills the presence role).
  - LAYOUT: per-shot entity count -> named positions (1=center, 2=left/right,
    3=thirds) -> bbox via LISA bbox_from_layout. (Positional heuristic stands in
    for the neural spatial planner; matches LISA's own layout approach.)

Generation stage: ported image-LISA (SDXL + native ip_adapter_masks), white-bg
anchors, complexity-adaptive sigma/scale, presence-aware per shot.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_lisa_planned.py \
    --story examples/user_story_001.json --out outputs/lisa/planned_story001
"""
import sys, os, argparse, gc, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import torch, yaml

from src.generation.lisa.build_anchors import build_all_anchors
from src.generation.lisa.layout_planner import create_layout_plan
from src.generation.lisa.lisa_pipeline import LISAPipeline

POS_BY_COUNT = {
    1: ["center"],
    2: ["left", "right"],
    3: ["left-third", "center-third", "right-third"],
}
BG_KEYWORDS = {
    "park": "a green park with trees", "office": "a modern office room",
    "street": "a city street", "beach": "a sandy beach", "kitchen": "a kitchen",
    "forest": "a forest", "sidewalk": "a sidewalk in front of a wall",
    "bench": "a park with a bench", "snow": "a snowy landscape",
}


def adaptive(n):
    if n <= 1:   return 20.0, 0.6
    if n == 2:   return 12.0, 0.7
    return 8.0, 0.8


def plan_presence(shots, entities):
    """Rule-based presence: entity present in a shot iff its name appears."""
    per_shot = []
    for txt in shots:
        toks = set(re.findall(r"[a-z]+", txt.lower()))
        present = [e for e in entities if e.lower() in toks]
        if not present:
            present = entities[:1]   # never empty
        per_shot.append(present)
    return per_shot


def infer_background(shots):
    blob = " ".join(shots).lower()
    for k, v in BG_KEYWORDS.items():
        if k in blob:
            return v
    return "a simple natural background"


def build_scenario(story):
    shots = story["shots"]
    refs = story.get("entity_references", {})
    entities = list(refs.keys()) or sorted(
        set(re.findall(r"[a-z]+", " ".join(shots).lower())) &
        {"cat", "dog", "bird", "horse", "rabbit", "sheep", "cow", "duck",
         "man", "woman", "child", "car", "bicycle"})
    eprompts = story.get("entity_prompts", {})
    presence = plan_presence(shots, entities)

    scenario = {
        "name": story.get("name", "planned_story"),
        "entities": [{"name": e, "prompt": eprompts.get(e, f"a {e}")} for e in entities],
        "background": {"name": "bg", "prompt": infer_background(shots)},
        "shots": [],
    }
    for txt, present in zip(shots, presence):
        pos = POS_BY_COUNT.get(len(present), POS_BY_COUNT[3])
        scenario["shots"].append({
            "description": txt,
            "entities": [{"name": e, "position": pos[i % len(pos)]}
                         for i, e in enumerate(present)],
        })
    return scenario, presence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", required=True)
    ap.add_argument("--config", default="configs/lisa_default.yaml")
    ap.add_argument("--out", default="outputs/lisa/planned")
    args = ap.parse_args()
    base = Path(__file__).parent.parent

    story = json.loads((base / args.story).read_text())
    scenario, presence = build_scenario(story)
    print("=== PLANNER OUTPUT ===")
    print("entities:", [e["name"] for e in scenario["entities"]])
    print("background:", scenario["background"]["prompt"])
    for i, sh in enumerate(scenario["shots"]):
        print(f"  shot{i}: present={[ (e['name'],e['position']) for e in sh['entities'] ]}")

    with open(base / args.config) as f:
        config = yaml.safe_load(f)
    config["evaluation"]["output_dir"] = str(base / args.out)
    config["anchors"]["save_dir"] = str(base / args.out / "anchor_bank")
    device = config["generation"]["device"]
    dtype = torch.float16 if config["generation"]["dtype"] == "float16" else torch.float32

    # Phase 1 anchors
    print("\n[Phase1] anchors...")
    from diffusers import StableDiffusionXLPipeline
    bp = StableDiffusionXLPipeline.from_pretrained(
        config["models"]["sdxl"], torch_dtype=dtype, variant="fp16", use_safetensors=True)
    bp.to(device); bp.enable_vae_slicing()
    anchor_bank = build_all_anchors(bp, scenario["entities"], scenario["background"], config)
    del bp; gc.collect(); torch.cuda.empty_cache()

    # Phase 2 layout (positions -> bbox)
    print("[Phase2] layout...")
    L = config["layout"]["latent_size"]
    layout_plan = create_layout_plan(scenario, latent_h=L, latent_w=L)

    # Phase 3+4 LISA generation (adaptive, presence-aware)
    print("[Phase3+4] LISA generation...")
    lisa = LISAPipeline(config); lisa.load_models()
    for i, shot in enumerate(layout_plan["shots"]):
        n = len(shot["entities"]); sig, ips = adaptive(n)
        r = lisa.generate_shot(anchor_bank, layout_plan, shot_index=i,
                               seed=config["generation"]["seed"] + i,
                               sigma_override=sig, ip_scale_override=ips)
        print(f"  shot{i}: {[e['name'] for e in shot['entities']]} -> {r['save_path']}")
    print(f"\nDone -> {base/args.out}")


if __name__ == "__main__":
    main()
