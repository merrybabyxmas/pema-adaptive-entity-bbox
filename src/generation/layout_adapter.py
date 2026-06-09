from src.utils.box_ops import cxcywh_to_xyxy
import numpy as np


def plan_to_layout(plan_output: dict) -> list[dict]:
    """Convert infer_bbox_plan output to per-shot layout dicts."""
    layouts = []
    for shot in plan_output["shots"]:
        layout = {
            "shot_id": shot["shot_id"],
            "prompt": shot["prompt"],
            "entities": [],
        }
        for entity, box in shot.get("boxes", {}).items():
            layout["entities"].append({
                "name": entity,
                "box_xyxy": box,  # already xyxy normalized
            })
        layouts.append(layout)
    return layouts


def draw_layout_on_image(image, layout: dict) -> "PIL.Image":
    """Draw predicted bboxes on a PIL image."""
    import PIL.Image
    import PIL.ImageDraw
    import PIL.ImageFont

    draw = PIL.ImageDraw.Draw(image)
    W, H = image.size
    colors = ["red", "blue", "green", "orange", "purple", "cyan"]

    for i, entity in enumerate(layout["entities"]):
        x1, y1, x2, y2 = entity["box_xyxy"]
        box = [x1 * W, y1 * H, x2 * W, y2 * H]
        color = colors[i % len(colors)]
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0] + 2, box[1] + 2), entity["name"], fill=color)

    return image
