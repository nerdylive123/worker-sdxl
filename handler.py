import os
import base64
import torch

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from schemas import INPUT_SCHEMA
from inference import ModelHandler, generate_images

torch.cuda.empty_cache()

# Initialize models
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

    # Generate images using the refactored function
    out_images, refresh = generate_images(MODELS, inp, generator)

    # Process and return results
    urls = _save_and_upload_images(out_images, job["id"])
    result = {"images": urls, "image_url": urls[0], "seed": seed}
    if refresh:
        result["refresh_worker"] = True
    return result

runpod.serverless.start({"handler": generate_image})
