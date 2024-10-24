import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from PIL import Image
import argparse
import torch
from torchvision import transforms
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoTokenizer
)
from diffusers import DDPMScheduler, AutoencoderKL
import numpy as np
from utils_mask import get_mask_location
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
import apply_net


WIDTH_TO_USE, HEIGHT_TO_USE = 384, 512 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    mask[binary_mask] = 255
    output_mask = Image.fromarray(mask)
    return output_mask


def initialize_pipeline():
    base_path = "yisol/IDM-VTON"

    unet = UNet2DConditionModel.from_pretrained(
        base_path,
        subfolder="unet",
        torch_dtype=torch.float16,
    )
    unet.requires_grad_(False)

    tokenizer_one = AutoTokenizer.from_pretrained(
        base_path,
        subfolder="tokenizer",
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        base_path,
        subfolder="tokenizer_2",
        use_fast=False,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

    text_encoder_one = CLIPTextModel.from_pretrained(
        base_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        base_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        base_path,
        subfolder="image_encoder",
        torch_dtype=torch.float16,
    )
    vae = AutoencoderKL.from_pretrained(
        base_path,
        subfolder="vae",
        torch_dtype=torch.float16,
    )

    
    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
        base_path,
        subfolder="unet_encoder",
        torch_dtype=torch.float16,
    )
    
    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)
    
    for model in [UNet_Encoder, image_encoder, vae, unet, text_encoder_one, text_encoder_two]:
        model.requires_grad_(False)
    
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    pipe = TryonPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
    )
    pipe.unet_encoder = UNet_Encoder
    pipe.to(device)

    return pipe, openpose_model, parsing_model, tensor_transform

def start_tryon(pipe, openpose_model, parsing_model, tensor_transform, 
               human_image_path, garment_image_path, garment_description,
               use_auto_mask, use_auto_crop, denoise_steps, seed, output_path):
    
    garm_img = Image.open(garment_image_path).convert("RGB").resize((WIDTH_TO_USE, HEIGHT_TO_USE))
    human_img_orig = Image.open(human_image_path).convert("RGB")
    
    if use_auto_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((WIDTH_TO_USE, HEIGHT_TO_USE))
    else:
        human_img = human_img_orig.resize((WIDTH_TO_USE, HEIGHT_TO_USE))

    
    if use_auto_mask:
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((WIDTH_TO_USE, HEIGHT_TO_USE))
    else:
        
        mask = pil_to_binary_mask(human_img_orig.resize((WIDTH_TO_USE, HEIGHT_TO_USE)))
    
    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)
    
    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
    
    args = apply_net.create_argument_parser().parse_args((
        'show', '../configs/densepose_rcnn_R_50_FPN_s1x.yaml', 
        '../yisol/IDM-VTON/densepose/model_final_162be9.pkl', 'dp_segm', '-v', 
        
        '--opts', 'MODEL.DEVICE', 'cpu'
    ))
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((WIDTH_TO_USE, HEIGHT_TO_USE))
    
    with torch.no_grad():
        if garment_description.strip():  
            prompt = f"model is wearing {garment_description}"
            prompt_c = f"a photo of {garment_description}"
        else:
            prompt = "model is wearing stylish clothing"          
            prompt_c = "a photo of stylish clothing"              

        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
            prompt,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )
        prompt_embeds_c, _, _, _ = pipe.encode_prompt(
            prompt_c,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=negative_prompt,
        )
        
        pose_img = tensor_transform(pose_img).unsqueeze(0).to(device, torch.float16)
        garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device, torch.float16)
        generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
        
        images = pipe(
            prompt_embeds=prompt_embeds.to(device, torch.float16),
            negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
            pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
            num_inference_steps=denoise_steps,
            generator=generator,
            strength=1.0,
            pose_img=pose_img.to(device, torch.float16),
            text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
            cloth=garm_tensor.to(device, torch.float16),
            mask_image=mask,
            image=human_img, 
            height=HEIGHT_TO_USE,
            width=WIDTH_TO_USE,
            ip_adapter_image=garm_img.resize((WIDTH_TO_USE, HEIGHT_TO_USE)),
            guidance_scale=2.0,
        )[0]
    
    if use_auto_crop:
        out_img = images[0].resize(crop_size)        
        human_img_orig.paste(out_img, (int(left), int(top)))    
        final_image = human_img_orig
    else:
        final_image = images[0]
    
    final_image.save(output_path)
    mask_gray.save(f"masked_{os.path.basename(output_path)}")
    print(f"Output saved to {output_path} and masked image saved to masked_{os.path.basename(output_path)}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="IDM-VITON CLI Try-On Tool")
    parser.add_argument("--human_image", type=str, required=True, help="Path to the human image")
    parser.add_argument("--garment_image", type=str, required=True, help="Path to the garment image")
    parser.add_argument(
    "--garment_description",
    type=str,
    required=False,  
    default="",      
    help="Description of the garment (e.g., 'Short Sleeve Round Neck T-shirt')",
    )
    parser.add_argument("--use_auto_mask", action="store_true", help="Use auto-generated mask")
    parser.add_argument("--use_auto_crop", action="store_true", help="Use auto-crop and resizing")
    parser.add_argument("--denoise_steps", type=int, default=20, help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="output.png", help="Path to save the output image")
    return parser.parse_args()

def main():
    args = parse_arguments()
    pipe, openpose_model, parsing_model, tensor_transform = initialize_pipeline()
    start_tryon(
        pipe=pipe,
        openpose_model=openpose_model,
        parsing_model=parsing_model,
        tensor_transform=tensor_transform,
        human_image_path=args.human_image,
        garment_image_path=args.garment_image,
        garment_description=args.garment_description,
        use_auto_mask=args.use_auto_mask,
        use_auto_crop=args.use_auto_crop,
        denoise_steps=args.denoise_steps,
        seed=args.seed,
        output_path=args.output
    )

if __name__ == "__main__":
    main()
