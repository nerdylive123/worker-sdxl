import torch
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image

from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

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

def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]

@torch.inference_mode()
def generate_images(model_handler, inp, generator):
    """
    Generate images using the appropriate pipeline based on input parameters.
    
    Args:
        model_handler: The ModelHandler instance with loaded models
        inp: Validated input parameters
        generator: Torch generator with seed
        
    Returns:
        tuple: (generated_images, refresh_flag)
    """
    # Choose scheduler
    model_handler.base.scheduler = make_scheduler(inp["scheduler"], model_handler.base.scheduler.config)
    if hasattr(model_handler.base_img2img, 'scheduler'):
        model_handler.base_img2img.scheduler = make_scheduler(inp["scheduler"], model_handler.base_img2img.scheduler.config)

    # If an init image is provided, use img2img mode
    if inp.get("image_url"):
        init_img = load_image(inp["image_url"]).convert("RGB")
        out_images = model_handler.base_img2img(
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
        out_images = model_handler.base(
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
        
    return out_images, refresh
