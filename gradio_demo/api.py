import sys
import os
import io
import logging
import tempfile
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError
import torch
from tryon_cli import initialize_pipeline, start_tryon
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools

logging.basicConfig(
    level=logging.ERROR,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("tryon_api")

app = FastAPI(
    title="idm vton api",
    description="Try on project",
    version="1.0"
)

try:
    pipe, openpose_model, parsing_model, tensor_transform = initialize_pipeline()
    logger.info("Pipeline initialized successfully.")
except Exception as e:
    logger.exception("Failed to initialize the pipeline.")
    raise e  

executor = ThreadPoolExecutor(max_workers=4)  

async def run_tryon_in_thread(*args, **kwargs):
    """
    Runs the blocking `start_tryon` function in a separate thread.
    """
    func = functools.partial(start_tryon, **kwargs)
    loop = asyncio.get_running_loop()
    logger.debug(f"Submitting start_tryon to executor with kwargs: {kwargs}")
    await loop.run_in_executor(executor, func)
    logger.debug("start_tryon completed.")

@app.post("/tryon/")
async def tryon_endpoint(
    human_image: UploadFile = File(..., description="Human image file (PNG or JPEG)"),
    garment_image: UploadFile = File(..., description="Garment image file (PNG or JPEG)"),
    garment_description: str = Form("", description="Description of the garment (optional)"),
    use_auto_mask: bool = Form(False, description="Use auto-generated mask"),
    use_auto_crop: bool = Form(False, description="Use auto-crop and resizing"),
    denoise_steps: int = Form(20, description="Number of denoising steps"),
    seed: int = Form(42, description="Random seed for reproducibility"),
):
    try:
        logger.info("Received a try-on request.")     

        if human_image.content_type not in ["image/png", "image/jpeg"]:
            logger.error(f"Invalid human image format: {human_image.content_type}")
            raise HTTPException(status_code=400, detail="Invalid human image format. Only PNG and JPEG are supported.")
        if garment_image.content_type not in ["image/png", "image/jpeg"]:
            logger.error(f"Invalid garment image format: {garment_image.content_type}")
            raise HTTPException(status_code=400, detail="Invalid garment image format. Only PNG and JPEG are supported.")

        human_image_bytes = await human_image.read()
        garment_image_bytes = await garment_image.read()
 
        with tempfile.TemporaryDirectory() as tmpdirname:
            human_img_path = os.path.join(tmpdirname, "human_image.png")
            garment_img_path = os.path.join(tmpdirname, "garment_image.png")
            output_path = os.path.join(tmpdirname, "output_tryon.png")
   
            logger.debug(f"Saving human image to {human_img_path}")
            with open(human_img_path, "wb") as f:
                f.write(human_image_bytes)
            logger.debug(f"Saving garment image to {garment_img_path}")
            with open(garment_img_path, "wb") as f:
                f.write(garment_image_bytes)

            logger.info("Running try-on process.")

            tryon_kwargs = {
                'pipe': pipe,
                'openpose_model': openpose_model,
                'parsing_model': parsing_model,
                'tensor_transform': tensor_transform,
                'human_image_path': human_img_path,
                'garment_image_path': garment_img_path,
                'garment_description': garment_description,
                'use_auto_mask': use_auto_mask,
                'use_auto_crop': use_auto_crop,
                'denoise_steps': denoise_steps,
                'seed': seed,
                'output_path': output_path
            }

            tryon_task = asyncio.create_task(run_tryon_in_thread(**tryon_kwargs))

            await tryon_task

            if not os.path.exists(output_path):
                logger.error("Output image not found.")
                raise HTTPException(status_code=500, detail="Try-on process failed to generate output image.")

            logger.debug(f"Reading output image from {output_path}")
            with open(output_path, "rb") as f:
                output_image = f.read()

            logger.info("Try-on process completed successfully.")
            logger.info("Returning the output image.")

            return StreamingResponse(io.BytesIO(output_image), media_type="image/png")

    except HTTPException as he:
        logger.error(f"HTTPException: {he.detail}")
        raise he
    except Exception as e:
        logger.exception("An error occurred during the try-on process.")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again later.")
