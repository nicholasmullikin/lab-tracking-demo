#!/usr/bin/env bash
# Idempotent install of LM-EEC (NeurIPS 2025 ego-exo correspondence, SAM 2.1 base-plus
# backbone) into its own venv at /home/nick/src/LM-EEC. Never touches the battle env.
#
# What it does (each step is skipped when already done):
#   1. clone juneyeeHu/LM-EEC and check out the pinned commit
#   2. uv venv (Python 3.10, the version in the upstream environment.yml)
#   3. torch 2.7.1 + torchvision 0.22.1 from the cu128 index (the RTX 5070 Ti is
#      Blackwell; the upstream cu118 pins cannot drive it; 2.7.1+cu128 is the version
#      already proven on this machine)
#   4. `pip install -e .` with SAM2_BUILD_CUDA=0 (the CUDA connected-components
#      extension is optional post-processing; building it needs nvcc + torch headers)
#   5. the undeclared imports the package needs at construction time
#      (timm, matplotlib, scikit-learn, networkx) plus the inference tool's
#      natsort / pycocotools / opencv
#   6. the two released checkpoints from the authors' Google Drive folder via gdown
#      (`ExoEgo_checkpoint.pt`, `EgoExo_checkpoint.pt`, 1.0 GB each)
#   7. a symlink to the existing SAM 2.1 base-plus checkpoint under ./checkpoint/
#      (the path the training config expects; inference does not read it)
#   8. a CPU-only verification: config parses, model constructs, checkpoint loads
#      with no missing / unexpected keys, image encoder runs on a dummy pair.
#
# Usage: scripts/install_lm_eec.sh [--skip-verify]
set -euo pipefail

LM_EEC_DIR="${LM_EEC_DIR:-/home/nick/src/LM-EEC}"
LM_EEC_COMMIT="${LM_EEC_COMMIT:-b37e50e50fd03ae8625e6100da37bad3dfeb6aa4}"
SAM21_CHECKPOINT_DIR="${SAM21_CHECKPOINT_DIR:-/home/nick/src/Grounded-SAM-2/checkpoints}"
GDRIVE_FOLDER="https://drive.google.com/drive/folders/1tc5HNWl0j7BcJE4uX0Bzb6PiYdlIWvXx"
SKIP_VERIFY=0
for arg in "$@"; do
  case "$arg" in
    --skip-verify) SKIP_VERIFY=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '[install_lm_eec] %s\n' "$*"; }

if [ ! -d "$LM_EEC_DIR/.git" ]; then
  log "cloning LM-EEC into $LM_EEC_DIR"
  git clone https://github.com/juneyeeHu/LM-EEC.git "$LM_EEC_DIR"
fi
cd "$LM_EEC_DIR"
if [ "$(git rev-parse HEAD)" != "$LM_EEC_COMMIT" ]; then
  log "checking out pinned commit $LM_EEC_COMMIT"
  git fetch --quiet origin
  git checkout --quiet "$LM_EEC_COMMIT"
fi

if [ ! -x .venv/bin/python ]; then
  log "creating venv (python 3.10)"
  uv venv --python 3.10 .venv
fi
PY=.venv/bin/python

if ! "$PY" -c 'import torch, torchvision; assert torch.__version__.startswith("2.7.1")' 2>/dev/null; then
  log "installing torch 2.7.1 / torchvision 0.22.1 (cu128)"
  uv pip install --python "$PY" "torch==2.7.1" "torchvision==0.22.1" \
    --index-url https://download.pytorch.org/whl/cu128
fi

if ! "$PY" -c 'import sam2, natsort, pycocotools, cv2' 2>/dev/null; then
  log "installing LM-EEC (editable, no CUDA extension) + inference tool deps"
  SAM2_BUILD_CUDA=0 uv pip install --python "$PY" -e . natsort pycocotools opencv-python-headless
fi

if ! "$PY" -c 'import timm, matplotlib, sklearn, networkx' 2>/dev/null; then
  log "installing undeclared imports (timm, matplotlib, scikit-learn, networkx)"
  uv pip install --python "$PY" timm matplotlib scikit-learn networkx
fi

CKPT_DIR="$LM_EEC_DIR/checkpoints/LM-EEC-checkpoint"
if [ ! -s "$CKPT_DIR/ExoEgo_checkpoint.pt" ] || [ ! -s "$CKPT_DIR/EgoExo_checkpoint.pt" ]; then
  log "downloading released checkpoints from Google Drive (2 x 1.0 GB)"
  mkdir -p "$LM_EEC_DIR/checkpoints"
  (cd "$LM_EEC_DIR/checkpoints" && uvx gdown --folder "$GDRIVE_FOLDER")
fi

mkdir -p "$LM_EEC_DIR/checkpoint"
if [ ! -e "$LM_EEC_DIR/checkpoint/sam2.1_hiera_base_plus.pt" ]; then
  log "linking SAM 2.1 base-plus checkpoint from $SAM21_CHECKPOINT_DIR"
  ln -s "$SAM21_CHECKPOINT_DIR/sam2.1_hiera_base_plus.pt" "$LM_EEC_DIR/checkpoint/sam2.1_hiera_base_plus.pt"
fi

if [ "$SKIP_VERIFY" -eq 0 ]; then
  log "CPU verification (CUDA hidden)"
  CUDA_VISIBLE_DEVICES="" "$PY" -W ignore - <<'EOF'
import time
import torch
t0 = time.time()
assert not torch.cuda.is_available(), "CUDA must stay hidden during the verification"
from sam2.build_sam import build_sam2_video_predictor_ego
pred = build_sam2_video_predictor_ego(
    "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "checkpoints/LM-EEC-checkpoint/ExoEgo_checkpoint.pt",
    device="cpu",
)
x = torch.zeros(1, 3, pred.image_size, pred.image_size)
with torch.inference_mode():
    ego_out, exo_out = pred.forward_image(x, x)
n_params = sum(p.numel() for p in pred.parameters()) / 1e6
print(
    f"[install_lm_eec] ok: device={pred.device} image_size={pred.image_size} "
    f"params={n_params:.1f}M fpn={[tuple(t.shape) for t in ego_out['backbone_fpn']]} "
    f"in {time.time() - t0:.1f}s"
)
EOF
fi
log "done ($LM_EEC_DIR @ $(git rev-parse --short HEAD))"
