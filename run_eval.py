import torch
from libs.base_utils import do_resize_content
from imagedream.ldm.util import (
    instantiate_from_config,
    get_obj_from_str,
)
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
from inference import generate3d
from huggingface_hub import hf_hub_download
import json
import argparse
import shutil
from model import CRM
import PIL
import rembg
import os
from pipelines import TwoStagePipeline
from run import expand_to_square, remove_background, do_resize_content,preprocess_image
from eval import calculate_metrics
import lpips
parser = argparse.ArgumentParser()
parser.add_argument(
    "--inputdir",
    type=str,
    default="examples/kunkun.webp",
    help="dir for input image",
)
parser.add_argument(
    "--scale",
    type=float,
    default=5.0,
)
parser.add_argument(
    "--step",
    type=int,
    default=50,
)
parser.add_argument(
    "--bg_choice",
    type=str,
    default="Auto Remove background",
    help="[Auto Remove background] or [Alpha as mask]",
)
parser.add_argument(
    "--outdir",
    type=str,
    default="pixel/",
)    
args = parser.parse_args()





img = Image.open(args.inputdir)
img = preprocess_image(img, args.bg_choice, 1.0, (127, 127, 127))
os.makedirs(args.outdir, exist_ok=True)
img.save(args.outdir+"preprocessed_image.png")

crm_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="CRM.pth")
specs = json.load(open("configs/specs_objaverse_total.json"))
model = CRM(specs).to("cuda")
model.load_state_dict(torch.load(crm_path, map_location = "cuda"), strict=False)

stage1_config = OmegaConf.load("configs/nf7_v3_SNR_rd_size_stroke.yaml").config
stage2_config = OmegaConf.load("configs/stage2-v2-snr.yaml").config
stage2_sampler_config = stage2_config.sampler
stage1_sampler_config = stage1_config.sampler

stage1_model_config = stage1_config.models
stage2_model_config = stage2_config.models

xyz_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="ccm-diffusion.pth")
pixel_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="pixel-diffusion.pth")
stage1_model_config.resume = pixel_path
stage2_model_config.resume = xyz_path


# sampler = 'DDIM'


# pipeline = TwoStagePipeline(
#     stage1_model_config,
#     stage2_model_config,
#     stage1_sampler_config,
#     stage2_sampler_config,
#     sampler = sampler
# )


# stage1_images = pipeline.call_image_multiview(img, scale=args.scale, step=args.step)
# refrence_image = stage1_images[-1]

# refrence_image.save(args.outdir+"refrence_image.png")

# np_imgs = np.concatenate(stage1_images, 1)
# Image.fromarray(np_imgs).save(args.outdir+"pixel_images.png")
# avoid loading the model multiple times
loss_fn_vgg = lpips.LPIPS(net="vgg")

def diffuse(img_dir, scale,outdir,pipeline):
    input_imgs = []
    for i in range(0, 5):
        img = Image.open(f"{img_dir}/{i}.jpg")
        img = preprocess_image(img, args.bg_choice, 1.0, (127, 127, 127))
        os.makedirs(outdir, exist_ok=True)
        img.save(outdir+f"preprocessed_image_{i}.png")
        input_imgs.append(img)
    
    stage1_images = pipeline.call_image_multiview(input_imgs[1], scale=scale, step=args.step)
    for i, image in enumerate(stage1_images):
        image.save(outdir+f"pixel_image_{i}.png")
    psnr = []
    ssim = []
    lpips = []
    pairs = [(1,5),(3,2),(2,3),(4,0)]
    for i, j in pairs:
        psnr.append(calculate_metrics(input_imgs[i], stage1_images[j],loss_fn_vgg)[0])
        ssim.append(calculate_metrics(input_imgs[i], stage1_images[j],loss_fn_vgg)[1])
        lpips.append(calculate_metrics(input_imgs[i], stage1_images[j],loss_fn_vgg)[2])
    print(f"PSNR: {psnr}")
    print(f"SSIM: {ssim}")
    print(f"LPIPS: {lpips}")
    psnr = np.array(psnr).mean()
    ssim = np.array(ssim).mean()
    lpips = np.array(lpips).mean()
    return psnr, ssim,lpips


def evaluate(sampler,scale):
    pipeline = TwoStagePipeline(
        stage1_model_config,
        stage2_model_config,
        stage1_sampler_config,
        stage2_sampler_config,
        sampler = sampler,
        # device = f"cuda:{int(scale)-1}"
    )
    base_output_dir = f"/root/CRM/eval/{sampler}/scale{scale}/"
    os.makedirs(base_output_dir, exist_ok=True)
    with open(f"{base_output_dir}metrics.txt", "w") as f:
        for i in range(1,20):
            psnrs = []
            ssims = []
            lpipss = []
            # loss_fn_vgg = lpips.LPIPS(net="vgg")
            imgdir = f"/root/CRM/GSO_extracted/{i}/thumbnails"
            output_dir = f"{base_output_dir}{i}/"
            psnr,ssim, lpips = diffuse(imgdir, scale, output_dir,pipeline)
            psnrs.append(psnr)
            ssims.append(ssim)
            lpipss.append(lpips)
            f.write(f"{i} {psnr} {ssim} {lpips} \n")
        f.write(f"Average PSNR: {np.array(psnrs).mean()} \n")
        f.write(f"Average SSIM: {np.array(ssims).mean()} \n")
        f.write(f"Average LPIPS: {np.array(lpipss).mean()} \n")

# from concurrent.futures import ThreadPoolExecutor
if __name__ == "__main__":
    # with ThreadPoolExecutor() as executor:
    for scale in range(2,8):
        scale = float(scale)
        for sampler in ['dpm-solver',"DDIM"]:
            evaluate(sampler,scale)
