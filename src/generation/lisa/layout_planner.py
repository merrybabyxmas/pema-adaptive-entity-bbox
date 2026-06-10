"""
Phase 2: Spatial Layout Planning.

Generates layout_plan.json with per-shot entity positions, bounding boxes, and prompts.
"""

import json
import os
from pathlib import Path


def create_layout_plan(
    scenario: dict,
    latent_h: int = 128,
    latent_w: int = 128,
    output_path: str = None,
) -> dict:
    """Create spatial layout plan from scenario definition.

    Args:
        scenario: {
            "name": str,
            "entities": [{"name": str, "prompt": str}, ...],
            "background": {"name": str, "prompt": str},
            "shots": [
                {
                    "description": str,
                    "entities": [
                        {"name": str, "position": "left"|"right"|"center"|...}
                    ]
                }
            ]
        }
        latent_h, latent_w: latent grid dimensions
        output_path: optional path to save JSON

    Returns:
        layout_plan dict
    """
    from src.generation.lisa.mask_utils import bbox_from_layout

    shots = []
    for i, shot_def in enumerate(scenario["shots"]):
        shot = {
            "shot_index": i,
            "description": shot_def["description"],
            "entities": [],
        }

        for ent_def in shot_def["entities"]:
            bbox = bbox_from_layout(ent_def["position"], latent_h, latent_w)
            shot["entities"].append({
                "name": ent_def["name"],
                "position": ent_def["position"],
                "bbox": list(bbox),
            })

        shots.append(shot)

    plan = {
        "scenario_name": scenario["name"],
        "latent_h": latent_h,
        "latent_w": latent_w,
        "entity_definitions": scenario["entities"],
        "background": scenario["background"],
        "shots": shots,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(plan, f, indent=2)
        print(f"  Layout plan saved to: {output_path}")

    return plan


def get_default_2entity_scenario() -> dict:
    """Default 2-entity test scenario for quick validation."""
    return {
        "name": "park_encounter",
        "entities": [
            {
                "name": "entity_A",
                "prompt": "a young blonde woman wearing a red dress",
            },
            {
                "name": "entity_B",
                "prompt": "a middle-aged man with dark hair wearing a black suit",
            },
        ],
        "background": {
            "name": "bg_park",
            "prompt": "a beautiful park with green trees and a walking path, golden hour lighting",
        },
        "shots": [
            {
                "description": "Both characters standing in a park, facing each other",
                "entities": [
                    {"name": "entity_A", "position": "left"},
                    {"name": "entity_B", "position": "right"},
                ],
            },
            {
                "description": "Both characters walking together on a park path",
                "entities": [
                    {"name": "entity_A", "position": "left"},
                    {"name": "entity_B", "position": "right"},
                ],
            },
            {
                "description": "Close-up of both characters having a conversation",
                "entities": [
                    {"name": "entity_A", "position": "left"},
                    {"name": "entity_B", "position": "right"},
                ],
            },
        ],
    }


def get_3entity_scenario() -> dict:
    """3-entity scenario for scaling test."""
    return {
        "name": "office_meeting",
        "entities": [
            {
                "name": "entity_A",
                "prompt": "a young woman with short black hair wearing a white blouse",
            },
            {
                "name": "entity_B",
                "prompt": "an older man with gray beard wearing a blue shirt",
            },
            {
                "name": "entity_C",
                "prompt": "a young man with brown hair wearing glasses and a green sweater",
            },
        ],
        "background": {
            "name": "bg_office",
            "prompt": "a modern office meeting room with glass walls and a white table",
        },
        "shots": [
            {
                "description": "Three colleagues in a meeting room discussion",
                "entities": [
                    {"name": "entity_A", "position": "left"},
                    {"name": "entity_B", "position": "center"},
                    {"name": "entity_C", "position": "right"},
                ],
            },
        ],
    }
