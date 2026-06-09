import sys; sys.path.insert(0,".")
import torch
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import AutoModel, AutoImageProcessor
dev="cuda"
pipe=StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
       torch_dtype=torch.float16, safety_checker=None).to(dev)
pipe.set_progress_bar_config(disable=True)
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
anchor=Image.open("outputs/generations/v13e/a0.5/anchors/anchor_dog_crop.png").convert("RGB")  # BLACK dog
outd=Path("outputs/generations/ipa_step1b"); outd.mkdir(parents=True, exist_ok=True)
anchor.save(outd/"anchor_BLACKdog.png")
proc=AutoImageProcessor.from_pretrained("facebook/dinov2-small")
dino=AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
@torch.no_grad()
def feat(im):
    x=proc(images=im,return_tensors="pt").to(dev); return torch.nn.functional.normalize(dino(**x).last_hidden_state[:,0],dim=-1).squeeze(0)
def sim(a,b): return float((feat(a)*feat(b)).sum())
print("BLACK-dog anchor → IP-Adapter SD1.5:")
for scale in [0.0, 0.7, 1.0]:
    pipe.set_ip_adapter_scale(scale)
    img=pipe(prompt="a dog sitting on pavement, photo", ip_adapter_image=anchor,
             num_inference_steps=30, guidance_scale=7.5,
             generator=torch.Generator(dev).manual_seed(0)).images[0]
    img.save(outd/f"gen_scale{scale}.png")
    print(f"  scale={scale} | DINOv2(gen,anchor)={sim(img,anchor):.3f}")
