from huggingface_hub import hf_hub_download

# download just the .safetensors file into your cache
# download_weights.py

local_path = hf_hub_download(
    repo_id="Lykon/dreamshaper-xl-v2-turbo",
    filename="DreamShaperXL_Turbo_v2_1.safetensors",
    cache_dir="/hf_cache",
)

#delete if exists
import os
if os.path.exists('/model_checkpoint_path.txt'):
    os.remove('/model_checkpoint_path.txt')

# persist the exact checkpoint location
with open('/model_checkpoint_path.txt', "w") as f:
    f.write(local_path)
    print("Checkpoint saved to", local_path)
print("Pipelines Downloaded")
