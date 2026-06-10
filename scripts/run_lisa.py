"""
Image-LISA (ported into adaptive_entity_bbox): SDXL + native IP-Adapter
ip_adapter_masks, Gaussian sum-to-1 region masks, white-bg anchors, complexity-
adaptive sigma/scale. Presence-aware multi-shot.

This is the higher-quality multi-shot generator brought over from LISA_code_686ba9b
(image path only). Layout positions here are hand-specified; our planner will
later feed inferred bboxes + presence in this same format.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_lisa.py --scenario dalmatian_cat
"""
import sys, os, argparse, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import torch, yaml

from src.generation.lisa.build_anchors import build_all_anchors
from src.generation.lisa.layout_planner import create_layout_plan
from src.generation.lisa.lisa_pipeline import LISAPipeline


def adaptive(n):
    """complexity-adaptive (sigma, ip_scale) by entity count (LISA run_10shot)."""
    if n <= 1:   return 20.0, 0.6
    if n == 2:   return 12.0, 0.7
    return 8.0, 0.8


# presence-aware multi-shot scenarios (distinctive entities + clean positions)
SCENARIOS = {
    "dalmatian_cat": {
        "name": "dalmatian_cat_story",
        "entities": [
            {"name": "dog", "prompt": "a dalmatian dog with black spots"},
            {"name": "cat", "prompt": "a black and white tuxedo cat"},
        ],
        "background": {"name": "bg_sidewalk",
                       "prompt": "a sidewalk in front of a brick wall, daytime"},
        "shots": [
            {"description": "the dalmatian sitting on a sidewalk",
             "entities": [{"name": "dog", "position": "center"}]},
            {"description": "the dalmatian and the tuxedo cat sitting together on a sidewalk",
             "entities": [{"name": "dog", "position": "left"},
                          {"name": "cat", "position": "right"}]},
            {"description": "the tuxedo cat sitting on a sidewalk",
             "entities": [{"name": "cat", "position": "center"}]},
        ],
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lisa_default.yaml")
    ap.add_argument("--scenario", default="dalmatian_cat")
    ap.add_argument("--out", default="outputs/lisa/dalmatian_cat")
    ap.add_argument("--self-attn-raan", action="store_true",
                    help="enable RAAN self-attention masking (anti-duplication)")
    args = ap.parse_args()

    base = Path(__file__).parent.parent
    with open(base / args.config) as f:
        config = yaml.safe_load(f)
    config["evaluation"]["output_dir"] = str(base / args.out)
    config["anchors"]["save_dir"] = str(base / args.out / "anchor_bank")
    if args.self_attn_raan:
        config.setdefault("conditioning", {}).setdefault("self_attn_control", {})["enabled"] = True
    device = config["generation"]["device"]
    dtype = torch.float16 if config["generation"]["dtype"] == "float16" else torch.float32
    scenario = SCENARIOS[args.scenario]

    # Phase 1: anchors (plain SDXL, white bg)
    print("[Phase1] anchors...")
    from diffusers import StableDiffusionXLPipeline
    base_pipe = StableDiffusionXLPipeline.from_pretrained(
        config["models"]["sdxl"], torch_dtype=dtype, variant="fp16", use_safetensors=True)
    base_pipe.to(device); base_pipe.enable_vae_slicing()
    anchor_bank = build_all_anchors(base_pipe, scenario["entities"], scenario["background"], config)
    del base_pipe; gc.collect(); torch.cuda.empty_cache()

    # Phase 2: layout (named position -> bbox)
    print("[Phase2] layout...")
    L = config["layout"]["latent_size"]
    layout_plan = create_layout_plan(scenario, latent_h=L, latent_w=L)

    # Phase 3+4: LISA generation, presence-aware + adaptive per shot
    print("[Phase3+4] LISA SDXL + ip_adapter_masks...")
    lisa = LISAPipeline(config); lisa.load_models()
    results = []
    for i, shot in enumerate(layout_plan["shots"]):
        n = len(shot["entities"])
        sig, ipscale = adaptive(n)
        r = lisa.generate_shot(anchor_bank, layout_plan, shot_index=i,
                               seed=config["generation"]["seed"] + i,
                               sigma_override=sig, ip_scale_override=ipscale)
        names = [e["name"] for e in shot["entities"]]
        print(f"  shot{i}: {names} (sigma={sig}, ip_scale={ipscale}) -> {r['save_path']}")
        results.append(r)
    print(f"\nDone. anchors+shots in {base/args.out}")


if __name__ == "__main__":
    main()
