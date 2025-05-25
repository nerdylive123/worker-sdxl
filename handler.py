import os
import base64

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoencoderKL,
)
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
    pipe.vae.half()
    return pipe

class ModelHandler:
    def __init__(self):
        self.base = None
        self.refiner = None
        self.load_models()

    def load_base(self):
        # Load VAE in half precision
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        # Load Base pipeline in half precision
        base_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
            local_files_only=True,
        ).to("cuda")
        base_pipe.enable_xformers_memory_efficient_attention()
        return to_fp16(base_pipe)

    def load_refiner(self):
        # Load VAE in half precision
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        # Load Refiner pipeline in half precision
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
            local_files_only=True,
        ).to("cuda")
        refiner_pipe.enable_xformers_memory_efficient_attention()
        return to_fp16(refiner_pipe)

    def load_models(self):
        self.base = self.load_base()
        self.refiner = self.load_refiner()

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

    # If an init image is provided, run only the refiner
    if inp.get("image_url"):
        init_img = load_image(inp["image_url"]).convert("RGB")
        out_images = MODELS.refiner(
            prompt=inp["prompt"],
            num_inference_steps=inp["refiner_inference_steps"],
            strength=inp["strength"],
            image=init_img,
            generator=generator,
        ).images
        refresh = True
    else:
        # 1) Generate float16 latents from the base pipeline
        latents = MODELS.base(
            prompt=inp["prompt"],
            negative_prompt=inp["negative_prompt"],
            height=inp["height"],
            width=inp["width"],
            num_inference_steps=inp["num_inference_steps"],
            guidance_scale=inp["guidance_scale"],
            denoising_end=inp["high_noise_frac"],
            output_type="latent",
            num_images_per_prompt=inp["num_images"],
            generator=generator,
        ).images

        # 2) Ensure latents are float16
        latents = latents.half()

        # 3) Refine those latents
        out_images = MODELS.refiner(
            prompt=inp["prompt"],
            num_inference_steps=inp["refiner_inference_steps"],
            strength=inp["strength"],
            image=latents,
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
