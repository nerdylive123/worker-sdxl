import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoencoderKL,
)
from huggingface_hub import snapshot_download


def get_diffusion_pipelines(cache_dir: str = "hf_cache"):
    """
    1) Snapshot each model repo to disk (so we never stream weights directly into GPU RAM).
    2) Load pipelines from the local folder.
    3) Enable CPU offload so layers migrate to CUDA only when they're used.
    """
    common_args = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "use_safetensors": True,
    }

    # 1) download/snapshot to disk
    base_dir = snapshot_download(
        repo_id="Lykon/dreamshaper-xl-v2-turbo",
        cache_dir=cache_dir,
        resume_download=True,
    )
    # vae_dir = snapshot_download(
    #     repo_id="madebyollin/sdxl-vae-fp16-fix",
    #     cache_dir=cache_dir,
    #     resume_download=True,
    # )

if __name__ == "__main__":
    get_diffusion_pipelines()
    print("Pipelines Downloaded")
