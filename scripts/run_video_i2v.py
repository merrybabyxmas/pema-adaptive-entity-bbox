"""
Video extension: per-shot keyframe -> multi-shot video.

CORE (ours) = the presence-aware bbox planner: it already produced per-shot
keyframes (presence + layout + occlusion-depth + identity) under a generation
dir (e.g. outputs/lisa/stories30_depth2). EVERYTHING ELSE (the actual video
synthesis) is the OFFICIAL CogVideoX-5b-I2V model via diffusers — no custom
generator. We animate each keyframe into a clip and concatenate the shots into
one multi-shot video per story.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_video_i2v.py \
    --keyframes outputs/lisa/stories30_depth2 \
    --stories examples/stories30_invocab.json \
    --out outputs/video/stories30_i2v [--limit N] [--only iv_18_tiger_dog,iv_29_rabbit_duck]
"""
import sys, os, argparse, gc, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import torch
from PIL import Image
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video

MODEL = "THUDM/CogVideoX-5b-I2V"
W, H = 720, 480  # CogVideoX-I2V native


def motion_prompt(story, shot):
    ents = " and ".join(e["prompt"] for e in story["entities"]
                        if e["name"] in shot["present"])
    bg = story.get("background", "")
    return (f"{ents} {bg}. Cinematic shot, the subjects move and breathe "
            f"naturally with gentle camera motion, high quality, detailed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyframes", default="outputs/lisa/stories30_depth2")
    ap.add_argument("--stories", default="examples/stories30_invocab.json")
    ap.add_argument("--out", default="outputs/video/stories30_i2v")
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated story names")
    args = ap.parse_args()

    base = Path(__file__).parent.parent
    kfdir = base / args.keyframes
    out = base / args.out; out.mkdir(parents=True, exist_ok=True)
    stories = json.loads((base / args.stories).read_text())
    if args.only:
        keep = set(args.only.split(","))
        stories = [s for s in stories if s["name"] in keep]
    if args.limit:
        stories = stories[:args.limit]
    print(f"[video] {len(stories)} stories via OFFICIAL {MODEL}")

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()          # fits in 24GB
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)

    for st in stories:
        sdir = kfdir / st["name"]
        odir = out / st["name"]; odir.mkdir(parents=True, exist_ok=True)
        all_frames = []
        n_shots = len(st["shots"])
        for s in range(n_shots):
            kf = sdir / f"shot_{s:03d}.png"
            if not kf.exists():
                print(f"  ! missing keyframe {kf}"); continue
            img = Image.open(kf).convert("RGB").resize((W, H))
            prompt = motion_prompt(st, st["shots"][s])
            gen = torch.Generator(device="cuda").manual_seed(42 + s)
            frames = pipe(image=img, prompt=prompt, num_frames=args.frames,
                          num_inference_steps=args.steps, guidance_scale=args.guidance,
                          generator=gen).frames[0]
            export_to_video(frames, str(odir / f"shot_{s:03d}.mp4"), fps=args.fps)
            all_frames.extend(frames)
            print(f"    {st['name']} shot{s} -> {len(frames)} frames", flush=True)
        if all_frames:
            export_to_video(all_frames, str(odir / "multishot.mp4"), fps=args.fps)
            print(f"  {st['name']} multishot -> {len(all_frames)} frames", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
