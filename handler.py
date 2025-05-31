import os
import base64

import torch
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
from diffusers.utils import load_image

from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from schemas import INPUT_SCHEMA

torch.cuda.empty_cache()

def to_fp16(pipe):
    """Convert all sub-modules of a Diffusers pipeline to float16."""
    pipe.unet.half()
    pipe.text_encoder.half()
    return pipe

class ModelHandler:
    # Centralized model name
    MODEL_NAME = "lykon/dreamshaper-xl-v2-turbo"
    
    def __init__(self):
        self.base = None
        self.base_img2img = None
        self.load_models()

    def load_base(self):
        # Load Base pipeline in half precision
        base_pipe = AutoPipelineForText2Image.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
            local_files_only=True,
        ).to("cuda")
        base_pipe.enable_xformers_memory_efficient_attention()
        return to_fp16(base_pipe)

    def load_base_img2img(self):
        # Load Base img2img pipeline in half precision
        base_img2img_pipe = AutoPipelineForText2Image.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
            local_files_only=True,
        ).to("cuda")
        base_img2img_pipe.enable_xformers_memory_efficient_attention()
        return to_fp16(base_img2img_pipe)

    def load_models(self):
        self.base = self.load_base()
        self.base_img2img = self.load_base_img2img()

MODELS = ModelHandler()

def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for idx, img in enumerate(images):
        path = os.path.join(f"/{job_id}", f"{idx}.png")
        img.save(path)
        if os.environ.get("BUCKET_ENDPOINT_URL"):
            url = rp_upload.upload_image(job_id, path)
        else:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                url = f"data:image/png;base64,{b64}"
        image_urls.append(url)
    rp_cleanup.clean([f"/{job_id}"])
    return image_urls

def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]

@torch.inference_mode()
def generate_image(job):
    # Debug logging
    print("[generate_image] RAW job dict:", job, flush=True)

    job_input = job.get("input", {})
    validated = validate(job_input, INPUT_SCHEMA)
    if "errors" in validated:
        return {"error": validated["errors"]}
    inp = validated["validated_input"]

    # Seed & generator
    seed = inp.get("seed") or int.from_bytes(os.urandom(2), "big")
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Choose scheduler
    MODELS.base.scheduler = make_scheduler(inp["scheduler"], MODELS.base.scheduler.config)
    if hasattr(MODELS.base_img2img, 'scheduler'):
        MODELS.base_img2img.scheduler = make_scheduler(inp["scheduler"], MODELS.base_img2img.scheduler.config)

    # If an init image is provided, use img2img mode
    if inp.get("image_url"):
        init_img = load_image(inp["image_url"]).convert("RGB")
        out_images = MODELS.base_img2img(
            prompt=inp["prompt"],
            negative_prompt=inp["negative_prompt"],
            num_inference_steps=inp["num_inference_steps"],
            guidance_scale=inp["guidance_scale"],
            strength=inp["strength"],
            image=init_img,
            generator=generator,
            num_images_per_prompt=inp["num_images"],
        ).images
        refresh = True
    else:
        # Generate images directly from the base pipeline
        out_images = MODELS.base(
            prompt=inp["prompt"],
            negative_prompt=inp["negative_prompt"],
            height=inp["height"],
            width=inp["width"],
            num_inference_steps=inp["num_inference_steps"],
            guidance_scale=inp["guidance_scale"],
            num_images_per_prompt=inp["num_images"],
            generator=generator,
        ).images
        refresh = False

    urls = _save_and_upload_images(out_images, job["id"])
    result = {"images": urls, "image_url": urls[0], "seed": seed}
    if refresh:
        result["refresh_worker"] = True
    return result

runpod.serverless.start({"handler": generate_image})
