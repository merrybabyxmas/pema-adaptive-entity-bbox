"""
v14 multi-shot with two presence patterns, on a method-suited pair
A = dalmatian dog, B = tuxedo cat (distinctive, well-separated).

  case1: A -> AB -> B
  case2: AB -> A -> B

Fixed per-entity anchors (SAM bg-masked, IP-Adapter-Plus) injected only into the
shots where the entity is present (presence matrix), localized to that shot's
bbox: solo shot = large centered box, pair shot = left/right boxes.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/make_multishot_cases.py --out outputs/generations/multishot_cases
"""
import sys, os, argparse, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionPipeline, StableDiffusionGLIGENPipeline
from transformers import AutoModel, AutoImageProcessor
from src.generation.pema_pipeline import GLIGEN_MODEL
from src.generation.entity_ip_adapter import (
    EntityIPController, extract_ip_adapter, entity_tokens, install_entity_ip,
)
from src.generation.anchor_segment import AnchorSegmenter

A = "a dalmatian dog"
B = "a tuxedo cat"
SOLO = [0.24, 0.16, 0.76, 0.95]
LEFT = [0.05, 0.28, 0.46, 0.92]
RIGHT = [0.54, 0.22, 0.96, 0.95]
SCENE = "sitting on a sidewalk in front of a brick wall, photo"
ACENTER = [0.24, 0.14, 0.76, 0.92]

CASES = {
    "case1_A_AB_B": [
        {"present": [A], "boxes": {A: SOLO}},
        {"present": [A, B], "boxes": {A: LEFT, B: RIGHT}},
        {"present": [B], "boxes": {B: SOLO}},
    ],
    "case2_AB_A_B": [
        {"present": [A, B], "boxes": {A: LEFT, B: RIGHT}},
        {"present": [A], "boxes": {A: SOLO}},
        {"present": [B], "boxes": {B: SOLO}},
    ],
}


def prompt_for(present):
    names = " and ".join(present)
    return f"{names} {SCENE}"


def draw(img, boxes):
    im = img.copy(); d = ImageDraw.Draw(im); W, H = im.size
    cols = {A: "red", B: "blue"}
    for n, b in boxes.items():
        c = cols.get(n, "green")
        d.rectangle([b[0]*W, b[1]*H, b[2]*W, b[3]*H], outline=c, width=3)
        d.text((b[0]*W+3, b[1]*H+3), n.split()[-1], fill=c)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/generations/multishot_cases")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    base = Path(__file__).parent.parent
    dev = "cuda"
    out = base / args.out; out.mkdir(parents=True, exist_ok=True)

    sd = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,
        safety_checker=None).to(dev)
    sd.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                       weight_name="ip-adapter-plus_sd15.bin")
    ip = extract_ip_adapter(sd)
    seg = AnchorSegmenter(device=dev)
    g = StableDiffusionGLIGENPipeline.from_pretrained(
        GLIGEN_MODEL, torch_dtype=torch.float16).to(dev)
    g.set_progress_bar_config(disable=True)

    # anchors (before installing IP)
    anchors = {}
    for k, ent in enumerate([A, B]):
        img = g(prompt=f"{ent}, full body, centered, plain neutral background, photo",
                gligen_phrases=[ent], gligen_boxes=[ACENTER],
                gligen_scheduled_sampling_beta=1.0, num_inference_steps=args.steps,
                guidance_scale=7.5, height=512, width=512,
                generator=torch.Generator(dev).manual_seed(args.seed + 700 + k)).images[0]
        img.save(out / f"anchor_{ent.split()[-1]}.png")
        anchors[ent] = (img, entity_tokens(ip, img, dev, obj_mask=seg.mask(img)))
    print("anchors done", flush=True)

    ctrl = EntityIPController(scale=args.scale, cfg=True, t_apply_below=1000.0,
                             feather=0.10, text_suppress=1.0)
    install_entity_ip(g.unet, ctrl, ip)

    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
    @torch.no_grad()
    def feat(im):
        x = proc(images=im, return_tensors="pt").to(dev)
        return torch.nn.functional.normalize(dino(**x).last_hidden_state[:, 0], dim=-1).squeeze(0)
    def sim(a, b): return float((feat(a)*feat(b)).sum())
    def crop(im, b):
        W, H = im.size
        return im.crop((int(b[0]*W), int(b[1]*H), int(b[2]*W), int(b[3]*H)))

    for cname, shots in CASES.items():
        cdir = out / cname; cdir.mkdir(exist_ok=True)
        regions = {A: [], B: []}   # (shot_id, region_img)
        for sid, shot in enumerate(shots):
            present, boxes = shot["present"], shot["boxes"]
            ctrl.set_active([(n, anchors[n][1], boxes[n]) for n in present])
            img = g(prompt=prompt_for(present), gligen_phrases=present,
                    gligen_boxes=[boxes[n] for n in present],
                    gligen_scheduled_sampling_beta=1.0, num_inference_steps=args.steps,
                    guidance_scale=7.5, height=512, width=512,
                    generator=torch.Generator(dev).manual_seed(args.seed + sid)).images[0]
            img.save(cdir / f"shot_{sid}.png")
            draw(img, boxes).save(cdir / f"shot_{sid}_bbox.png")
            for n in present:
                regions[n].append((sid, crop(img, boxes[n])))
        # metrics + sequence montage
        msg = []
        for ent in [A, B]:
            rs = regions[ent]
            anc = anchors[ent][0]
            af = sum(sim(r, anc) for _, r in rs)/len(rs)
            cs = "n/a"
            if len(rs) > 1:
                cs = sum(sim(rs[i][1], rs[j][1]) for i, j in itertools.combinations(range(len(rs)), 2)) \
                     / (len(rs)*(len(rs)-1)//2)
                cs = f"{cs:.3f}"
            msg.append(f"{ent.split()[-1]}: shots{[s for s,_ in rs]} cross={cs} vs-anchor={af:.3f}")
        print(f"[{cname}] " + " | ".join(msg), flush=True)
        # montage row of shots
        W = H = 200; bar = 16
        m = Image.new("RGB", (3*W, H+bar), "white"); d = ImageDraw.Draw(m)
        for sid in range(len(shots)):
            m.paste(Image.open(cdir/f"shot_{sid}_bbox.png").convert("RGB").resize((W, H)), (sid*W, bar))
            d.text((sid*W+3, 3), f"shot{sid}: {'+'.join(p.split()[-1] for p in shots[sid]['present'])}", fill="black")
        m.save(out / f"_{cname}_seq.png")
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
