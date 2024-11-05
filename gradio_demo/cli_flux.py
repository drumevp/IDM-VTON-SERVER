## TODO: Must install diffusers===0.31.0 to use
## Needs at least 40GB of RAM

import sys
import os
import logging
import psutil
from pathlib import Path
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from PIL import Image
import argparse
import torch
from torchvision import transforms
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
# from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
# from src.unet_hacked_tryon import UNet2DConditionModel
from diffusers import FluxTransformer2DModel, UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoTokenizer,
    BlipProcessor,
    BlipForConditionalGeneration
)
from diffusers import DDPMScheduler, AutoencoderKL
import numpy as np
from utils_mask import get_mask_location
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
import apply_net

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

PROJECT_ROOT = parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, str(PROJECT_ROOT))

def log_memory_usage(step_description):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.info(f"{step_description} - Memory usage: {mem_info.rss / (1024 ** 2):.2f} MB")

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    mask[binary_mask] = 255
    output_mask = Image.fromarray(mask)

    return output_mask

def initialize_pipeline():
    logging.info("Initializing pipeline...")
    base_path = "yisol/IDM-VTON"

    # Log memory usage
    log_memory_usage("Before loading UNet")

    unet_path = "/home/ubuntu/IDM-VTON-SERVER/test-flux/idmvtonflux/transformers"
    unet_encoder_path = "/home/ubuntu/IDM-VTON-SERVER/test-flux/idmvtonflux/unet_encoder"
    vae_path = "/home/ubuntu/IDM-VTON-SERVER/test-flux/idmvtonflux/vae"
    image_encoder_path = "/home/ubuntu/IDM-VTON-SERVER/test-flux/idmvtonflux/image_encoder"
    text_encoder_path = "/home/ubuntu/IDM-VTON-SERVER/test-flux/idmvtonflux/text_encoder"
    text_encoder_2_path = "/home/ubuntu/IDM-VTON-SERVER/test-flux/idmvtonflux/text_encoder_2"

    start_time = time.time()
    try:
        unet = FluxTransformer2DModel.from_pretrained(
            unet_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True,
        )
        unet.requires_grad_(False)
        logging.info("UNet loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading UNet: {e}")
        sys.exit(1)
    logging.info(f"UNet loading time: {time.time() - start_time:.2f} seconds")
    log_memory_usage("After loading UNet")

    logging.info("Loading tokenizers...")
    try:
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
        logging.info("Tokenizers loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading tokenizers: {e}")
        sys.exit(1)
    log_memory_usage("After loading tokenizers")

    logging.info("Loading noise scheduler...")
    try:
        noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
        logging.info("Noise scheduler loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading noise scheduler: {e}")
        sys.exit(1)
    log_memory_usage("After loading noise scheduler")

    logging.info("Loading text and image encoders...")
    try:
        text_encoder_one = CLIPTextModel.from_pretrained(
            text_encoder_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True,
        )
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            text_encoder_2_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True,
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True,
        )
        logging.info("Text and image encoders loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading text and image encoders: {e}")
        sys.exit(1)
    log_memory_usage("After loading text and image encoders")

    logging.info("Loading VAE...")
    try:
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True,
        )
        logging.info("VAE loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading VAE: {e}")
        sys.exit(1)
    log_memory_usage("After loading VAE")

    logging.info("Loading UNet Encoder...")
    try:
        UNet_Encoder = UNet2DConditionModel.from_pretrained(
            unet_encoder_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True,
        )
        logging.info("UNet Encoder loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading UNet Encoder: {e}")
        sys.exit(1)
    log_memory_usage("After loading UNet Encoder")

    logging.info("Initializing parsing and openpose models...")
    try:
        parsing_model = Parsing(0)
        openpose_model = OpenPose(0)
        logging.info("Parsing and OpenPose models initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing parsing and OpenPose models: {e}")
        sys.exit(1)
    log_memory_usage("After initializing parsing and OpenPose models")

    # Initialize BLIP for image captioning
    logging.info("Initializing BLIP model for garment description...")
    try:
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        logging.info("BLIP model initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing BLIP model: {e}")
        sys.exit(1)

    # Disable gradient computation for models
    for model in [UNet_Encoder, image_encoder, vae, unet, text_encoder_one, text_encoder_two, blip_model]:
        model.requires_grad_(False)

    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    logging.info("Setting up TryonPipeline...")
    try:
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
        logging.info("TryonPipeline set up successfully.")
    except Exception as e:
        logging.error(f"Error setting up TryonPipeline: {e}")
        sys.exit(1)
    log_memory_usage("After setting up TryonPipeline")

    pipe.unet_encoder = UNet_Encoder
    pipe.to(device)
    logging.info("Pipeline moved to device.")

    return pipe, openpose_model, parsing_model, tensor_transform, blip_model, blip_processor

def start_tryon(
    pipe,
    openpose_model,
    parsing_model,
    tensor_transform,
    human_image_path,
    garment_image_path,
    garment_description,
    use_auto_mask,
    use_auto_crop,
    denoise_steps,
    seed,
    output_path,
    width,
    height,
    guidance_scale,
    strength,
    should_use_clip,
    blip_model,
    blip_processor,
    body_part, # upper_body, lower_body, dresses
):
    logging.info("Starting try-on process...")
    log_memory_usage("Before loading images")

    try:
        garm_img = Image.open(garment_image_path).convert("RGB").resize((width, height))
        human_img_orig = Image.open(human_image_path).convert("RGB")
        logging.info("Images loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading images: {e}")
        sys.exit(1)

    if use_auto_crop:
        logging.info("Auto-cropping enabled.")
        width_orig, height_orig = human_img_orig.size
        target_width = int(min(width_orig, height_orig * (width / height)))
        target_height = int(min(height_orig, width_orig * (height / width)))
        left = (width_orig - target_width) / 2
        top = (height_orig - target_height) / 2
        right = (width_orig + target_width) / 2
        bottom = (height_orig + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((width, height))
        logging.info(f"Image cropped to size: {crop_size}")
    else:
        human_img = human_img_orig.resize((width, height))
        logging.info("Auto-cropping disabled.")

    log_memory_usage("After processing images")

    if use_auto_mask:
        logging.info("Generating auto mask...")
        try:
            keypoints = openpose_model(human_img.resize((width, height)))
            model_parse, _ = parsing_model(human_img.resize((width, height)))
            mask, mask_gray = get_mask_location('hd', body_part, model_parse, keypoints)
            mask = mask.resize((width, height))
            logging.info("Auto mask generated successfully.")
        except Exception as e:
            logging.error(f"Error generating auto mask: {e}")
            sys.exit(1)
    else:
        mask = pil_to_binary_mask(human_img_orig.resize((width, height)))
        logging.info("Using manual mask.")

    log_memory_usage("After generating mask")

    try:
        mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img)
        mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)
        logging.info("Mask gray image created.")
    except Exception as e:
        logging.error(f"Error processing mask: {e}")
        sys.exit(1)

    log_memory_usage("After processing mask")

    logging.info("Preparing pose image...")
    try:
        human_img_arg = _apply_exif_orientation(human_img.resize((width, height)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

        densepose_ckpts_path = os.path.join(PROJECT_ROOT, 'yisol/IDM-VTON/densepose/model_final_162be9.pkl')

        args = apply_net.create_argument_parser().parse_args((
            'show', '../configs/densepose_rcnn_R_50_FPN_s1x.yaml',
            densepose_ckpts_path, 'dp_segm', '-v',
            '--opts', 'MODEL.DEVICE', 'cuda'
        ))
        pose_img = args.func(args, human_img_arg)
        pose_img = pose_img[:, :, ::-1]
        pose_img = Image.fromarray(pose_img).resize((width, height))
        logging.info("Pose image prepared successfully.")
    except Exception as e:
        logging.error(f"Error preparing pose image: {e}")
        sys.exit(1)

    log_memory_usage("After preparing pose image")

    logging.info("Starting inference...")
    try:
        with torch.no_grad():
            if should_use_clip:
                logging.info("Generating garment description using BLIP...")
                try:
                    inputs = blip_processor(garm_img, return_tensors="pt").to(device)
                    outputs = blip_model.generate(**inputs)
                    garment_description = blip_processor.decode(outputs[0], skip_special_tokens=True)
                    logging.info(f"Generated garment description: {garment_description}")
                except Exception as e:
                    logging.error(f"Error generating garment description with BLIP: {e}")
                    sys.exit(1)
            else:
                logging.info("Using provided garment description.")

            if garment_description.strip():
                prompt = f"model is wearing {garment_description}"
                prompt_c = f"a photo of {garment_description}"
            else:
                prompt = "model is wearing stylish clothing"
                prompt_c = "a photo of stylish clothing"

            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

            logging.info("Encoding prompts...")
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
            logging.info("Prompts encoded.")

            pose_img_tensor = tensor_transform(pose_img).unsqueeze(0).to(device, torch.float16)
            garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device, torch.float16)
            generator = torch.Generator(device).manual_seed(seed) if seed is not None else None

            log_memory_usage("Before pipeline inference")

            images = pipe(
                prompt_embeds=prompt_embeds.to(device, torch.float16),
                negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                num_inference_steps=denoise_steps,
                generator=generator,
                strength=strength,
                pose_img=pose_img_tensor,
                text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                cloth=garm_tensor,
                mask_image=mask,
                image=human_img,
                height=height,
                width=width,
                ip_adapter_image=garm_img.resize((width, height)),
                guidance_scale=guidance_scale,
            )[0]
            logging.info("Inference completed.")
            log_memory_usage("After pipeline inference")
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        sys.exit(1)

    try:
        if use_auto_crop:
            out_img = images[0].resize(crop_size)
            human_img_orig.paste(out_img, (int(left), int(top)))
            final_image = human_img_orig
        else:
            final_image = images[0]

        final_image.save(output_path)
        mask_gray.save(f"masked_{os.path.basename(output_path)}")
        logging.info(f"Output saved to {output_path} and masked image saved to masked_{os.path.basename(output_path)}")
    except Exception as e:
        logging.error(f"Error saving output images: {e}")
        sys.exit(1)

    logging.info("Try-on process completed successfully.")

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
    parser.add_argument("--use_auto_mask", action="store_true", default=True, help="Use auto-generated mask")
    parser.add_argument("--use_auto_crop", action="store_true", default=False, help="Use auto-crop and resizing")
    parser.add_argument("--denoise_steps", type=int, default=30, help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="output.png", help="Path to save the output image")
    parser.add_argument("--width", type=int, default=768, help="Width of the output image")
    parser.add_argument("--height", type=int, default=1024, help="Height of the output image")
    parser.add_argument("--guidance_scale", type=float, default=2.0, help="Guidance scale for the model")
    parser.add_argument("--strength", type=float, default=1.0, help="Strength parameter for the model")
    parser.add_argument("--should_use_clip", action="store_false", default=False, help="Use CLIP to generate garment description")
    parser.add_argument("--body_part", default="upper_body", help="Set which parts of the body the garment covers. Options: upper_body, lower_body, dresses")

    return parser.parse_args()

def main():
    args = parse_arguments()
    log_memory_usage("At script start")
    pipe, openpose_model, parsing_model, tensor_transform, blip_model, blip_processor = initialize_pipeline()
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
        output_path=args.output,
        width=args.width,
        height=args.height,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        should_use_clip=args.should_use_clip,
        blip_model=blip_model,
        blip_processor=blip_processor,
        body_part=args.body_part,
    )
    log_memory_usage("At script end")

if __name__ == "__main__":
    main()
