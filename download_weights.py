from huggingface_hub import hf_hub_download

# download just the .safetensors file into your cache
# download_weights.py

import os
import time
from huggingface_hub import hf_hub_download
import pprint

cache_dir = "/hf_cache"

def tree(path):
    """Recursively print a directory tree."""
    for root, dirs, files in os.walk(path):
        level = root.replace(path, "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root) or path}/")
        subindent = " " * 2 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

print("=== BEFORE DOWNLOAD ===")
if os.path.exists(cache_dir):
    tree(cache_dir)
else:
    print(f"{cache_dir} does not exist yet")

# your download call
local_path = hf_hub_download(
    repo_id="Lykon/dreamshaper-xl-v2-turbo",
    filename="DreamShaperXL_Turbo_v2_1.safetensors",
    cache_dir=cache_dir,
)

print("\nDownloaded file path:", local_path)
print("\n=== AFTER DOWNLOAD ===")
tree(cache_dir)

# sanity check for parent directory too
parent = os.path.dirname(local_path)
print("\nContents of parent snapshot dir:", parent)
print(os.listdir(parent))

#delete if exists
import os
if os.path.exists('/model_checkpoint_path.txt'):
    os.remove('/model_checkpoint_path.txt')

# persist the exact checkpoint location
with open('/model_checkpoint_path.txt', "w") as f:
    f.write(local_path)
    print("Checkpoint saved to", local_path)
print("Pipelines Downloaded")
