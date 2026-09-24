#!/bin/bash
# Bootstrap a fresh National Compute MI355X node (Ubuntu, user ubuntu) for
# spherical-pigeon Stage E. Idempotent. Usage: bash bootstrap.sh
set -euo pipefail
ROOT=/mnt/nvme/sp
mkdir -p $ROOT && cd $ROOT
echo "== hardware"; (rocm-smi --showproductname 2>/dev/null | grep -i "card series" | head -8) || true
ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null || echo unknown); echo "ROCm $ROCM_VER"
echo "== python env"
if [ ! -x venv/bin/python ]; then
  (sudo apt-get install -y -q python3-venv rsync tmux >/dev/null 2>&1 || true)
  python3 -m venv venv
fi
. venv/bin/activate
pip install -q --upgrade pip
if ! python -c "import torch" 2>/dev/null; then
  MAJMIN=$(echo "$ROCM_VER" | cut -d. -f1,2)
  for idx in "rocm${MAJMIN}" rocm7.0 rocm6.4; do
    echo "trying torch wheel index $idx"
    if pip install -q torch --index-url https://download.pytorch.org/whl/$idx; then break; fi
  done
fi
pip install -q numpy pandas h3 scipy wandb tqdm transformers huggingface_hub pillow rasterio
python - <<'PY'
import torch
print('torch', torch.__version__, 'gpu available', torch.cuda.is_available(),
      'count', torch.cuda.device_count(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
PY
mkdir -p $ROOT/spherical-pigeon/{data/energy,saved_models/energy,data/logs}
echo BOOTSTRAP_OK
