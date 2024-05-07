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





# img = Image.open(args.inputdir)
img = Image.open("/root/CRM/GSO_extracted/1/thumbnails/1.jpg")
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


sampler = 'DDIM'


pipeline = TwoStagePipeline(
    stage1_model_config,
    stage2_model_config,
    stage1_sampler_config,
    stage2_sampler_config,
    sampler = sampler
)
img1 = Image.open("/root/CRM/GSO_extracted/1/thumbnails/2.jpg")
img1= preprocess_image(img1, args.bg_choice, 1.0, (127, 127, 127))
additional_input_images = [img1]
# additional_input_images = None
additional_input_positions = [3]
# additional_input_positions = None

stage1_images = pipeline.call_image_multiview(img, scale=args.scale, step=args.step, additional_input_images=additional_input_images, additional_input_positions=additional_input_positions)
# refrence_image = stage1_images[-1]

# refrence_image.save(args.outdir+"refrence_image.png")

np_imgs = np.concatenate(stage1_images, 1)
Image.fromarray(np_imgs).save(args.outdir+"pixel_images.png")

stage1_images_no_additional_input = pipeline.call_image_multiview(img, scale=args.scale, step=args.step)
np_imgs = np.concatenate(stage1_images_no_additional_input, 1)
Image.fromarray(np_imgs).save(args.outdir+"pixel_images_no_additional_input.png")
# for i,img in enumerate(stage1_images):
#     img.save(args.outdir+f"pixel_images{i}.png")