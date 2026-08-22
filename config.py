"""Paths and constants. Standalone — this project shares no code with the
PIGEON repository; baseline comparisons against PIGEON run in a separate
clone of the official release.
"""

import os

# Image encoder (HuggingFace id). geolocal/StreetCLIP is the PIGEON authors'
# public caption-pretrained checkpoint, same ViT-L/14-336 architecture as the
# OpenAI default — use it so every ablation row shares one encoder.
CLIP_MODEL = os.environ.get('PIGEON_CLIP_MODEL', 'openai/clip-vit-large-patch14-336')
CLIP_EMBED_DIM = 1024

# OSV-5M (Astruc et al., CVPR 2024)
OSV5M_HF_REPO = 'osv5m/osv5m'
OSV5M_ROOT = 'data/osv5m'
METADATA_PATH_OSV = 'data/osv5m/metadata_osv5m.csv'
IMAGE_PATH_OSV = 'data/osv5m/images'

# Rasters (see get_rasters.sh)
KOPPEN_GEIGER_PATH = 'data/rasters/koppen_geiger/Beck_KG_V1_present_0p0083.tif'
GHSL_PATH = 'data/rasters/pop_density/GHS_POP_E2020_GLOBE_R2022A_54009_1000_V1_0.tif'

# Benchmarks: name -> {"meta": csv path, "images": dir}
BENCHMARKS = 'data/benchmarks/benchmarks.json'
