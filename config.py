import os
from transformers import TrainingArguments

# Image data & metadata paths

# OpenAI's pretrained implementation. Override via PIGEON_CLIP_MODEL to swap
# encoders, e.g. PIGEON_CLIP_MODEL=geolocal/StreetCLIP (the authors' public
# caption-pretrained checkpoint, same ViT-L/14-336 architecture).
CLIP_MODEL = os.environ.get('PIGEON_CLIP_MODEL', 'openai/clip-vit-large-patch14-336')
CLIP_EMBED_DIM = 1024

### StreetView
METADATA_PATH = 'data/data_duels.csv'
PRETRAIN_METADATA_PATH = 'data/data_pretrain.csv'
IMAGE_PATH = 'data/streetview_outputs_cropped'
INPUT_PATH = 'data/streetview_outputs'
IMAGE_PATH_2 = 'data/streetview_part_2_data'

### YFCC
METADATA_PATH_YFCC = 'data/data_yfcc_augmented_non_contaminated.csv'
PRETRAIN_METADATA_PATH_YFCC = 'data/data_yfcc_augmented_non_contaminated.csv'
IMAGE_PATH_YFCC = 'data/images_mp_16/jpgs'

### Landmarks
METADATA_PATH_LANDMARKS = 'data/data_landmarks_aug.csv'
IMAGE_PATH_LANDMARKS = 'data/benchmarks/google_landmark/jpgs'

### OSV-5M (Astruc et al., CVPR 2024) — shared data spine for the PIGEON
### retrain control and the energy model. See dataset_creation/osv5m.py.
OSV5M_HF_REPO = 'osv5m/osv5m'
OSV5M_ROOT = 'data/osv5m'
METADATA_PATH_OSV = 'data/osv5m/metadata_osv5m.csv'
IMAGE_PATH_OSV = 'data/osv5m/images'
GEOCELL_PATH_OSV = 'data/geocells_osv5m.csv'
DATASET_PATH_OSV = 'data/hf_OSV5M'
SCALER_PATH_OSV = 'saved_models/scaler/regression_osv5m.scaler'
PROTO_PATH_OSV = 'data/data_prototypes_osv5m.csv'
PROTO_MODEL_OSV_PATH = 'saved_models/refiner/proto_osv5m.refiner'
PRETRAINED_CLIP_OSV = 'saved_models/StreetCLIP_osv5m.model'
CLIP_PRETRAINED_HEAD_OSV = 'saved_models/OSV5M_head.model'

# Political boundaries
COUNTRY_PATH = 'data/geocells/countries.geojson'
ADMIN_1_PATH = 'data/geocells/admin_1.geojson'
ADMIN_2_PATH = 'data/geocells/admin_2.geojson'

# Geocell creation
MIN_CELL_SIZE = 1000 # (PIGEOTTO), 30 (PIGEON)
MAX_CELL_SIZE = 2000 # (PIGEOTTO), 60 (PIGEON)

# Geocells path
GEOCELL_PATH = 'data/geocells_2203.csv' # PIGEON
GEOCELL_PATH_YFCC = 'data/geocells_yfcc.csv' # PIGEOTTO

# Scaler Path
SCALER_PATH = 'saved_models/scaler/regression.scaler'
SCALER_PATH_YFCC = 'saved_models/scaler/regression_yfcc.scaler'

# Geodata augmentation paths
WORLD_CITIES = 'data/benchmarks/gws15k/worldcities.csv'
GADM_PATH = 'data/gadm/gadm_410-levels.gpkg'
GHSL_PATH = 'data/pop_density/GHS_POP_E2020_GLOBE_R2022A_54009_1000_V1_0.tif'
WORLDCLIM_SAVE_PATH = 'data/worldclim'
SRTM_SAVE_PATH = 'data/elevation/'
KOPPEN_GEIGER_PATH = 'data/koppen_geiger/Beck_KG_V1_present_0p0083.tif'
DRIVING_SIDE_PATH = 'data/driving_side/countries_driving_side.json'

# Geoguessr formula
DECAY_CONSTANT = 1492.7

# Haversine smoothing constant
LABEL_SMOOTHING_CONSTANT = 65 # (PIGEOTTO), 75 (PIGEON)
LABEL_SMOOTHING_MONTHS = 0.3

# Models
CURRENT_SAVE_PATH = 'saved_models/WorldCLIP_head_landmarks.model'

### StreetView (PIGEON)
PRETRAINED_CLIP = 'saved_models/StreetviewCLIP.model'
CLIP_PRETRAINED_HEAD = 'saved_models/New_Base_smooth_avg_MT_Geo_SV.model'

### YFCC & Landmarks (PIGEOTTO)
PRETRAINED_CLIP_YFCC = 'saved_models/WorldCLIP.model'
CLIP_PRETRAINED_HEAD_YFCC = 'saved_models/WorldCLIP_head.model' # PIGEOTTO prediction head
CLIP_PRETRAINED_HEAD_YFCC_LANDMARKS = 'saved_models/WorldCLIP_head_landmarks.model' # PIGEOTTO prediction head with landmarks

# Embedding
EMBED_BATCH_SIZE_PER_GPU = 512

# Cluster Refinement Model

### StreetView
PROTO_PATH = 'data/data_prototypes_2203.csv'
DATASET_PATH = 'data/hf_SVCLIP_2203'
PROTO_MODEL_PATH = 'saved_models/refiner/proto.refiner'

### YFCC
PROTO_PATH_YFCC = 'data/data_prototypes_YFCC.csv'
DATASET_PATH_YFCC = 'data/hf_YFCC'
PROTO_MODEL_YFCC_PATH = 'saved_models/refiner/proto_YFCC.refiner'

### Landmarks
PROTO_PATH_LANDMARKS = 'data/data_prototypes_landmarks.csv'
DATASET_PATH_LANDMARKS = 'data/hf_landmarks'
PROTO_MODEL_LANDMARKS_PATH = 'saved_models/refiner/proto_landmarks.refiner'

# Benchmark Eval Paths
BENCHMARKS = 'data/benchmarks/benchmarks.json'

# OSV-5M rides the PIGEOTTO (--yfcc) code path with its artifact paths
# rebound here, so no module outside config.py needs a third dataset branch.
# run.py sets PIGEON_DATASET=osv5m when invoked with --osv; every module reads
# these constants from config, so rebinding at import time covers them all.
if os.environ.get('PIGEON_DATASET', '').lower() == 'osv5m':
    METADATA_PATH_YFCC = METADATA_PATH_OSV
    PRETRAIN_METADATA_PATH_YFCC = METADATA_PATH_OSV
    IMAGE_PATH_YFCC = IMAGE_PATH_OSV
    GEOCELL_PATH_YFCC = GEOCELL_PATH_OSV
    DATASET_PATH_YFCC = DATASET_PATH_OSV
    SCALER_PATH_YFCC = SCALER_PATH_OSV
    PROTO_PATH_YFCC = PROTO_PATH_OSV
    PROTO_MODEL_YFCC_PATH = PROTO_MODEL_OSV_PATH
    PRETRAINED_CLIP_YFCC = PRETRAINED_CLIP_OSV
    CLIP_PRETRAINED_HEAD_YFCC = CLIP_PRETRAINED_HEAD_OSV

# Training arguments --> RUN ON 4 GPUs
TRAIN_ARGS = TrainingArguments(
    output_dir='saved_models',
    remove_unused_columns=False,
    per_device_train_batch_size=256, # 1024 (256 (Batch), 4 (Cores), 1 (Accumulation))
    per_device_eval_batch_size=256,
    num_train_epochs=1000,
    evaluation_strategy='epoch',
    eval_steps=1,
    save_strategy='epoch',
    save_steps=1,
    learning_rate=2e-5,
    logging_steps=1,
    gradient_accumulation_steps=1,
    load_best_model_at_end=True,
    seed=330
)

# Pretrain arguments for PIGEOTTO --> RUN ON 4 A100 GPUs
PRETAIN_ARGS_YFCC = TrainingArguments(
        output_dir='saved_models/pretrained_yfcc',
        overwrite_output_dir = True,
        do_train=True,
        do_eval=True,
        evaluation_strategy='steps',
        eval_steps=50,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=8, # 12 for 3 GPUs
        learning_rate=5e-07, # was 1e-06 before
        weight_decay=0.001, # CHANGED
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-06,
        max_grad_norm=1.0,
        num_train_epochs=4, # 20 before
        max_steps=-1,
        lr_scheduler_type = 'linear',
        warmup_ratio = 0.02,
        logging_first_step = False,
        logging_steps=1,
        save_strategy='steps',
        save_steps=50,
        seed=42,
        dataloader_drop_last=True,
        run_name=None,
        adafactor=False,
        report_to='tensorboard',
        skip_memory_metrics=True,
        resume_from_checkpoint=None,
    )

# Pretrain arguments for PIGEON --> RUN ON 4 A100 GPUs
PRETAIN_ARGS = TrainingArguments(
        output_dir='saved_models/pretrained',
        overwrite_output_dir = True,
        do_train=True,
        do_eval=True,
        evaluation_strategy='steps',
        eval_steps=50,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=8, # 8 for 4 GPUs
        learning_rate=1e-06,
        weight_decay=0.001,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-06,
        max_grad_norm=1.0,
        num_train_epochs=20,
        max_steps=-1,
        lr_scheduler_type = 'linear',
        warmup_ratio = 0.2,
        logging_first_step = False,
        logging_steps=1,
        save_strategy='steps',
        save_steps=50,
        seed=42,
        dataloader_drop_last=True,
        run_name=None,
        adafactor=False,
        report_to='tensorboard',
        skip_memory_metrics=True,
        resume_from_checkpoint=None,
    )