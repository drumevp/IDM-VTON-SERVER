## TODO: Must install diffusers===0.31.0 to use
## Needs at least 40GB of RAM

import sys
import os
import io
import logging
import tempfile
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from cli_flux import initialize_pipeline, start_tryon
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
    pipe, openpose_model, parsing_model, tensor_transform, blip_model, blip_processor = initialize_pipeline()
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

@app.post("/tryon")
async def tryon_endpoint(
    human_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    garment_description: str = Form(""),
    use_auto_mask: bool = Form(True),
    use_auto_crop: bool = Form(True),
    denoise_steps: int = Form(30),
    seed: int = Form(42),
    width: int = Form(768),
    height: int = Form(1024),
    guidance_scale: float = Form(2.0),
    strength: float = Form(1.0),
    should_use_clip: bool = Form(False),
    body_part: str = Form("upper_body"),
):
    try:
        logger.info("Received a try-on request.")

        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_human_file_path = os.path.join(tmpdirname, "human_image.png")
            temp_garment_file_path = os.path.join(tmpdirname, "garment_image.png")
            output_path = os.path.join(tmpdirname, "output_tryon.png")

            with open(temp_human_file_path, "wb") as temp_human_file:
                temp_human_file.write(await human_image.read())

            with open(temp_garment_file_path, "wb") as temp_garment_file:
                temp_garment_file.write(await garment_image.read())

            logger.info("Running try-on process.")

            tryon_kwargs = {
                'pipe': pipe,
                'openpose_model': openpose_model,
                'parsing_model': parsing_model,
                'tensor_transform': tensor_transform,
                'human_image_path': temp_human_file_path,
                'garment_image_path': temp_garment_file_path,
                'garment_description': garment_description,
                'use_auto_mask': use_auto_mask,
                'use_auto_crop': use_auto_crop,
                'denoise_steps': denoise_steps,
                'seed': seed,
                'output_path': output_path,
                'width': width,
                'height': height,
                'guidance_scale': guidance_scale,
                'strength': strength,
                'should_use_clip': should_use_clip,
                'blip_model': blip_model,
                'blip_processor': blip_processor,
                'body_part': body_part,
            }

            await run_tryon_in_thread(**tryon_kwargs)

            if not os.path.exists(output_path):
                logger.error("Output image not found.")
                raise HTTPException(status_code=500, detail="Try-on process failed to generate output image.")

            logger.info("Try-on process completed successfully.")
            logger.info("Returning the output image.")
            with open(output_path, "rb") as output_file:
                output_image = output_file.read()

            return StreamingResponse(io.BytesIO(output_image), media_type="image/png")

    except HTTPException as he:
        logger.error(f"HTTPException: {he.detail}")
        raise he
    except Exception as e:
        logger.exception("An error occurred during the try-on process.")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again later.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
