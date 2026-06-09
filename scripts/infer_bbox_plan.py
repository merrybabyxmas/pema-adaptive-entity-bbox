import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
import numpy as np
from pathlib import Path
import yaml

from src.model.bbox_planner import build_model
from src.model.embeddings import CLIPTextEncoder
from src.lm_planner.prompts import build_plan_prompt
from src.lm_planner.parser import parse_plan_output, normalize_plan
from src.lm_planner.validator import validate_plan, build_presence_matrix, compute_states
from src.data.schema import STATE2ID, RELATION2ID
from src.utils.box_ops import cxcywh_to_xyxy
from src.utils.io import load_json, save_json
from src.utils.logging import get_logger

logger = get_logger("infer")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def rule_based_plan(shot_prompts: list[str]) -> dict:
    """Fallback rule-based planner when LM is not available."""
    import re
    common_entities = ["cat", "dog", "bird", "horse", "person", "man", "woman",
                       "child", "car", "bicycle", "rabbit", "cow", "sheep"]
    all_found = []
    per_shot = []
    for prompt in shot_prompts:
        tokens = re.findall(r'\b\w+\b', prompt.lower())
        found = [e for e in common_entities if e in tokens]
        per_shot.append(found)
        all_found.extend(found)

    entities = list(dict.fromkeys(all_found)) or ["entity"]
    shots = []
    for i, (prompt, active) in enumerate(zip(shot_prompts, per_shot)):
        if not active:
            active = entities[:1]
        shots.append({
            "shot_id": i,
            "background": "scene",
            "active_entities": active,
            "relations": [],
            "focus": active[0] if active else "none",
        })
    return {"entities": entities, "shots": shots}


def plan_to_tensors(plan: dict, max_shots=5, max_entities=5, device="cuda"):
    entities = plan["entities"][:max_entities]
    E = len(entities)
    S = len(plan["shots"])
    S_use = min(S, max_shots)
    ent2idx = {e: i for i, e in enumerate(entities)}

    P = build_presence_matrix({"entities": entities, "shots": plan["shots"][:S_use]})
    states = compute_states(P, entities)

    presence = np.zeros((max_shots, max_entities), dtype=np.int64)
    state_ids = np.zeros((max_shots, max_entities), dtype=np.int64)
    relation_ids = np.zeros((max_shots, max_entities, max_entities), dtype=np.int64)

    presence[:S_use, :E] = P
    for s in range(S_use):
        for ei, e in enumerate(entities):
            from src.data.schema import STATE2ID
            state_ids[s, ei] = STATE2ID.get(states[s][ei], 0)

    for s, shot in enumerate(plan["shots"][:S_use]):
        for rel in shot.get("relations", []):
            if len(rel) == 3:
                subj, rel_type, obj = rel
                if subj in ent2idx and obj in ent2idx:
                    si, oi = ent2idx[subj], ent2idx[obj]
                    if si < max_entities and oi < max_entities:
                        relation_ids[s, si, oi] = RELATION2ID.get(rel_type, 0)

    shot_prompts = [shot["prompt"] for shot in plan["shots"][:S_use]]
    shot_prompts += [""] * (max_shots - len(shot_prompts))
    entity_names = entities + [""] * (max_entities - E)

    return {
        "shot_prompts": [shot_prompts],
        "entity_names": [entity_names],
        "presence": torch.from_numpy(presence).unsqueeze(0).to(device),
        "state_ids": torch.from_numpy(state_ids).unsqueeze(0).to(device),
        "relation_ids": torch.from_numpy(relation_ids).unsqueeze(0).to(device),
        "num_shots": S_use,
        "num_entities": E,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/runs/bbox_planner_v1/checkpoints/best.pt")
    parser.add_argument("--input", default="examples/user_story_001.json")
    parser.add_argument("--out", default="outputs/eval/user_story_001_boxes.json")
    parser.add_argument("--max-shots", type=int, default=5)
    parser.add_argument("--max-entities", type=int, default=5)
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = base / args.checkpoint
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg_path = ckpt_path.parent.parent / "config.yaml"
    cfg = load_config(str(cfg_path))

    encoder = CLIPTextEncoder(model_name="ViT-B-32", pretrained="openai", freeze=True).to(device)
    model_cfg = cfg["model"]
    model_cfg["d_text"] = encoder.d_out
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(f"Model loaded from epoch {ckpt.get('epoch', '?')}")

    user_input = load_json(str(base / args.input))
    shot_prompts = user_input["shots"]

    # rule-based plan (or LM plan if available)
    plan = rule_based_plan(shot_prompts)
    # attach prompts to shots
    for i, shot in enumerate(plan["shots"]):
        shot["prompt"] = shot_prompts[i] if i < len(shot_prompts) else ""

    tensors = plan_to_tensors(plan, args.max_shots, args.max_entities, device)
    S_use = tensors["num_shots"]
    E_use = tensors["num_entities"]
    entities = plan["entities"][:E_use]

    with torch.no_grad():
        shot_emb = encoder.encode_batch_shots(tensors["shot_prompts"], device)
        entity_emb = encoder.encode_batch_entities(tensors["entity_names"], device)
        pred_boxes = model(shot_emb.float(), entity_emb.float(),
                           tensors["state_ids"], tensors["presence"], tensors["relation_ids"])
        pred_xyxy = cxcywh_to_xyxy(pred_boxes).clamp(0, 1)

    presence_np = tensors["presence"][0].cpu().numpy()
    result = {
        "entities": entities,
        "presence": presence_np[:S_use, :E_use].tolist(),
        "shots": [],
    }
    for s in range(S_use):
        shot_out = {
            "shot_id": s,
            "prompt": shot_prompts[s] if s < len(shot_prompts) else "",
            "boxes": {},
        }
        for ei, e in enumerate(entities):
            if presence_np[s, ei] == 1:
                box = pred_xyxy[0, s, ei].cpu().tolist()
                box = [round(v, 4) for v in box]
                shot_out["boxes"][e] = box
        result["shots"].append(shot_out)

    out_path = base / args.out
    save_json(result, str(out_path))
    logger.info(f"Inference result saved to {out_path}")

    for shot in result["shots"]:
        logger.info(f"Shot {shot['shot_id']}: {shot['prompt']}")
        for e, box in shot["boxes"].items():
            logger.info(f"  {e}: {box}")


if __name__ == "__main__":
    main()
