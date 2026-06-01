#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# EvidenceLens GPU Environment Setup
# Target GPUs: NVIDIA L4 / NVIDIA B200
# Conda source:
# /blue/ruogu.fang/qixuan.wu/conda/etc/profile.d/conda.sh
# ============================================================

PROJECT_ROOT="/orange/ruogu.fang/qixuan/EvidenceLens"
CONDA_SH="/blue/ruogu.fang/qixuan.wu/conda/etc/profile.d/conda.sh"
ENV_NAME="evidencelens_l4_b200"
PYTHON_VERSION="3.11"

cd "$PROJECT_ROOT"

echo "============================================================"
echo "[0] Basic system check"
echo "============================================================"

hostname || true
which python || true
which conda || true
nvidia-smi || true

echo ""
echo "============================================================"
echo "[1] Load conda"
echo "============================================================"

if [ ! -f "$CONDA_SH" ]; then
  echo "ERROR: conda.sh not found at:"
  echo "$CONDA_SH"
  exit 1
fi

source "$CONDA_SH"

echo "Conda path:"
which conda
conda --version

echo ""
echo "============================================================"
echo "[2] Create / reset conda env"
echo "============================================================"

# Remove old env only if user explicitly sets RESET_ENV=1
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  if [ "${RESET_ENV:-0}" = "1" ]; then
    echo "Removing existing env: $ENV_NAME"
    conda env remove -n "$ENV_NAME" -y
  else
    echo "Env already exists: $ENV_NAME"
    echo "Set RESET_ENV=1 bash setup_l4_b200_env.sh to rebuild from scratch."
  fi
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

echo "Python:"
which python
python --version

echo ""
echo "============================================================"
echo "[3] Upgrade core packaging tools"
echo "============================================================"

python -m pip install --upgrade pip setuptools wheel packaging ninja

echo ""
echo "============================================================"
echo "[4] Install PyTorch CUDA build"
echo "============================================================"

# Use CUDA 12.8 wheels for L4/B200 compatibility.
# This is usually the safest path for B200/Blackwell on Linux.
python -m pip install --upgrade \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

echo ""
echo "============================================================"
echo "[5] Install EvidenceLens core libraries"
echo "============================================================"

python -m pip install --upgrade \
  datasets \
  huggingface_hub \
  transformers \
  accelerate \
  evaluate \
  tokenizers \
  sentencepiece \
  safetensors \
  peft \
  trl \
  bitsandbytes \
  einops \
  scipy \
  scikit-learn \
  pandas \
  numpy \
  tqdm \
  jsonlines \
  pydantic \
  pyyaml \
  python-dotenv \
  requests \
  beautifulsoup4 \
  lxml \
  openai \
  anthropic \
  matplotlib \
  plotly \
  seaborn \
  jupyterlab \
  ipykernel \
  pytest \
  black \
  ruff

echo ""
echo "============================================================"
echo "[6] Install medical-imaging / scientific utilities"
echo "============================================================"

python -m pip install --upgrade \
  monai \
  nibabel \
  SimpleITK \
  nilearn \
  scikit-image \
  opencv-python-headless \
  medpy

echo ""
echo "============================================================"
echo "[7] Register Jupyter kernel"
echo "============================================================"

python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

echo ""
echo "============================================================"
echo "[8] Write environment activation helper"
echo "============================================================"

cat > activate_env.sh <<EOF
#!/usr/bin/env bash
source "$CONDA_SH"
conda activate "$ENV_NAME"
cd "$PROJECT_ROOT"
EOF

chmod +x activate_env.sh

echo ""
echo "============================================================"
echo "[9] Write verification script"
echo "============================================================"

mkdir -p scripts outputs/debug runs/slurm

cat > scripts/verify_l4_b200_env.py <<'PY'
import os
import sys
import json
import platform
from pathlib import Path

report = {}

def add(key, value):
    report[key] = value
    print(f"{key}: {value}")

add("python", sys.version)
add("platform", platform.platform())
add("executable", sys.executable)

# Core imports
imports = [
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "datasets",
    "accelerate",
    "evaluate",
    "pandas",
    "numpy",
    "sklearn",
    "jsonlines",
    "tqdm",
    "monai",
    "nibabel",
    "SimpleITK",
    "nilearn",
    "skimage",
    "cv2",
    "matplotlib",
    "plotly",
    "openai",
]

missing = []
for name in imports:
    try:
        mod = __import__(name)
        add(f"import_{name}", "OK")
    except Exception as e:
        missing.append((name, repr(e)))
        add(f"import_{name}", f"FAILED: {repr(e)}")

import torch

add("torch_version", torch.__version__)
add("torch_cuda_version", torch.version.cuda)
add("cuda_available", torch.cuda.is_available())
add("cuda_device_count", torch.cuda.device_count())

gpu_info = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        item = {
            "index": i,
            "name": props.name,
            "major": props.major,
            "minor": props.minor,
            "total_memory_gb": round(props.total_memory / 1024**3, 2),
        }
        gpu_info.append(item)
        print(f"GPU {i}: {item}")

    # Small GPU tensor test
    x = torch.randn(2048, 2048, device="cuda")
    y = torch.randn(2048, 2048, device="cuda")
    z = x @ y
    torch.cuda.synchronize()
    add("gpu_matmul_test", f"OK shape={tuple(z.shape)} dtype={z.dtype}")

    # Mixed precision test
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        z2 = x @ y
    torch.cuda.synchronize()
    add("gpu_amp_fp16_test", f"OK shape={tuple(z2.shape)} dtype={z2.dtype}")
else:
    add("gpu_matmul_test", "SKIPPED: CUDA not available")

report["gpu_info"] = gpu_info
report["missing_imports"] = missing

out = Path("outputs/debug/l4_b200_env_verification.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")

if missing:
    print("\nMISSING IMPORTS:")
    for name, err in missing:
        print(f"- {name}: {err}")
    raise SystemExit(1)

print(f"\n[OK] Environment verification written to {out}")
PY

echo ""
echo "============================================================"
echo "[10] Run verification"
echo "============================================================"

python scripts/verify_l4_b200_env.py | tee outputs/debug/l4_b200_env_verification_stdout.txt

echo ""
echo "============================================================"
echo "DONE"
echo "============================================================"
echo "Activate later with:"
echo "source /orange/ruogu.fang/qixuan/EvidenceLens/activate_env.sh"
echo ""
echo "Verification report:"
echo "/orange/ruogu.fang/qixuan/EvidenceLens/outputs/debug/l4_b200_env_verification.json"
