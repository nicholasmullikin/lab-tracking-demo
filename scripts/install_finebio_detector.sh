#!/usr/bin/env bash
# Idempotent, CPU-only install of the FineBio object detector (Yagi et al., IJCV 2025):
# MMDetection 3.x with the authors' DINO / Deformable DETR configs and released weights,
# in its own venv at /home/nick/src/finebio-detector. Never touches the battle env.
#
# Why CPU: the RTX 5070 Ti (Blackwell, sm_120) is driven only by torch 2.7.1+cu128 on this
# machine, no prebuilt mmcv wheel exists for that torch, and the GPU is shared with the SAM3
# tracker queue whose guard refuses to start beside another model process. DINO's
# MultiScaleDeformableAttention falls back to the pure-PyTorch path off CUDA, so inference
# needs no CUDA extension at all.
#
# What it does (each step is skipped when already done):
#   1. uv venv (Python 3.10)
#   2. torch 2.1.2 + torchvision 0.16.2 from the PyTorch CPU index (the torch line for which
#      OpenMMLab publishes CPU mmcv 2.x wheels)
#   3. mmengine 0.10.7, numpy<2 (mmcv 2.1.0 is compiled against numpy 1.x)
#   4. mmcv 2.1.0 with CPU ops: the OpenMMLab prebuilt CPU wheel for torch 2.1 (the same
#      artefact `mim install mmcv` resolves to); if that wheel does not import, build the PyPI
#      sdist from source with MMCV_WITH_OPS=1 and no CUDA
#   5. mmdetection cloned at v3.3.0 (the last release accepting mmcv<2.2.0), editable install
#   6. the FineBio object_detection files fetched at a pinned aistairc/FineBio commit
#      (two configs + their README), copied into mmdetection/configs/{dino,deformable_detr}/
#      as the authors' README instructs
#   7. the released weights from the authors' Google Drive links (the README's 2025-12-15
#      update; the older finebio.s3.abci.ai host no longer resolves) via gdown, sha256 recorded
#      in checkpoints/SHA256SUMS
#   8. a CPU-only verification: init_detector + inference_detector on a blank image
#
# Usage: scripts/install_finebio_detector.sh [--skip-verify] [--skip-weights]
set -euo pipefail

FINEBIO_DET_DIR="${FINEBIO_DET_DIR:-/home/nick/src/finebio-detector}"
MMDET_COMMIT="${MMDET_COMMIT:-44ebd17b145c2372c4b700bfb9cb20dbd28ab64a}" # tag v3.3.0
FINEBIO_COMMIT="${FINEBIO_COMMIT:-cb8d16ef13c7c9901418c13bcbaa50a3bdf3a2c3}" # aistairc/FineBio main, 2026-09-21
TORCH_VERSION=2.1.2
TORCHVISION_VERSION=0.16.2
MMCV_VERSION=2.1.0
MMENGINE_VERSION=0.10.7
MMCV_WHEEL_URL="https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/mmcv-${MMCV_VERSION}-cp310-cp310-manylinux1_x86_64.whl"
FINEBIO_RAW="https://raw.githubusercontent.com/aistairc/FineBio/${FINEBIO_COMMIT}/object_detection"
DINO_GDRIVE_ID="1VseplD0tPLJ89mzD5IGbCa16_gooNqXy"
DDETR_GDRIVE_ID="1U390LjTByULtgD6XlecWrHPAlU9AjH_W"
SKIP_VERIFY=0
SKIP_WEIGHTS=0
for arg in "$@"; do
  case "$arg" in
    --skip-verify) SKIP_VERIFY=1 ;;
    --skip-weights) SKIP_WEIGHTS=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '[install_finebio_detector] %s\n' "$*"; }

mkdir -p "$FINEBIO_DET_DIR"/{checkpoints,finebio_configs}
cd "$FINEBIO_DET_DIR"

if [ ! -x .venv/bin/python ]; then
  log "creating venv (python 3.10)"
  uv venv --python 3.10 .venv
fi
PY="$FINEBIO_DET_DIR/.venv/bin/python"

if ! "$PY" -c "import torch, torchvision; assert torch.__version__.startswith('${TORCH_VERSION}+cpu')" 2>/dev/null; then
  log "installing torch ${TORCH_VERSION}+cpu / torchvision ${TORCHVISION_VERSION}+cpu"
  uv pip install --python "$PY" "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url https://download.pytorch.org/whl/cpu
fi

if ! "$PY" -c "import mmengine, numpy; assert mmengine.__version__ == '${MMENGINE_VERSION}' and numpy.__version__ < '2'" 2>/dev/null; then
  log "installing mmengine ${MMENGINE_VERSION}, numpy<2"
  uv pip install --python "$PY" "mmengine==${MMENGINE_VERSION}" "numpy<2"
fi

mmcv_ok() {
  "$PY" - <<'EOF' 2>/dev/null
import mmcv
from mmcv.ops import multi_scale_deform_attn, nms  # both need the compiled _ext
assert mmcv.__version__ == "2.1.0", mmcv.__version__
EOF
}
if ! mmcv_ok; then
  log "installing mmcv ${MMCV_VERSION} (prebuilt CPU wheel for torch 2.1)"
  wheel="$FINEBIO_DET_DIR/$(basename "$MMCV_WHEEL_URL")"
  if [ ! -s "$wheel" ]; then
    curl -fsSL --retry 3 -o "$wheel.part" "$MMCV_WHEEL_URL" && mv "$wheel.part" "$wheel"
  fi
  uv pip install --python "$PY" "$wheel" "numpy<2" || true
  if ! mmcv_ok; then
    log "prebuilt wheel did not import; building mmcv ${MMCV_VERSION} from the PyPI sdist (CPU ops, no CUDA)"
    uv pip uninstall --python "$PY" mmcv || true
    uv pip install --python "$PY" setuptools wheel ninja
    MMCV_WITH_OPS=1 FORCE_CUDA=0 CUDA_HOME= MAX_JOBS="$(nproc)" \
      uv pip install --python "$PY" --no-build-isolation --no-binary mmcv "mmcv==${MMCV_VERSION}" "numpy<2"
  fi
  mmcv_ok || { log "mmcv ${MMCV_VERSION} still does not import with ops; stopping"; exit 1; }
fi

if [ ! -d mmdetection/.git ]; then
  log "cloning mmdetection into $FINEBIO_DET_DIR/mmdetection"
  git clone --quiet https://github.com/open-mmlab/mmdetection.git mmdetection
fi
if [ "$(git -C mmdetection rev-parse HEAD)" != "$MMDET_COMMIT" ]; then
  log "checking out mmdetection $MMDET_COMMIT (v3.3.0)"
  git -C mmdetection fetch --quiet origin
  git -C mmdetection checkout --quiet "$MMDET_COMMIT"
fi
if ! "$PY" -c "import mmdet; assert mmdet.__version__ == '3.3.0', mmdet.__version__" 2>/dev/null; then
  log "installing mmdetection (editable) + inference deps"
  # mmdet's setup.py imports torch.utils.cpp_extension, which on torch 2.1 still imports
  # pkg_resources (removed in setuptools 80), so the build cannot be isolated from the venv.
  uv pip install --python "$PY" "setuptools<80" wheel
  uv pip install --python "$PY" --no-build-isolation -e ./mmdetection "numpy<2"
fi

for f in dino-4scale_r50_8xb2-12e_finebio.py deformable-detr-refine-twostage_r50_16xb2-50e_finebio.py README.md; do
  if [ ! -s "finebio_configs/$f" ]; then
    log "fetching FineBio object_detection/$f at $FINEBIO_COMMIT"
    curl -fsSL --retry 3 -o "finebio_configs/$f.part" "$FINEBIO_RAW/$f" && mv "finebio_configs/$f.part" "finebio_configs/$f"
  fi
done
(cd finebio_configs && sha256sum dino-4scale_r50_8xb2-12e_finebio.py deformable-detr-refine-twostage_r50_16xb2-50e_finebio.py > SHA256SUMS)
# The authors' README: drop each config next to the MMDetection config it inherits from.
for pair in "dino-4scale_r50_8xb2-12e_finebio.py:dino" \
            "deformable-detr-refine-twostage_r50_16xb2-50e_finebio.py:deformable_detr"; do
  f="${pair%%:*}"; sub="${pair##*:}"
  if ! cmp -s "finebio_configs/$f" "mmdetection/configs/$sub/$f"; then
    log "placing $f into mmdetection/configs/$sub/"
    cp "finebio_configs/$f" "mmdetection/configs/$sub/$f"
  fi
done

if [ "$SKIP_WEIGHTS" -eq 0 ]; then
  for pair in "dino.pth:$DINO_GDRIVE_ID" "deformable-detr.pth:$DDETR_GDRIVE_ID"; do
    f="${pair%%:*}"; id="${pair##*:}"
    if [ ! -s "checkpoints/$f" ]; then
      log "downloading $f from the authors' Google Drive (id $id)"
      (cd checkpoints && uvx gdown -O "$f" "https://drive.google.com/file/d/$id/view?usp=sharing")
    fi
  done
  (cd checkpoints && sha256sum dino.pth deformable-detr.pth > SHA256SUMS)
fi

if [ "$SKIP_VERIFY" -eq 0 ]; then
  log "CPU verification (CUDA hidden): DINO config + weights, one blank 1333x800 image"
  CUDA_VISIBLE_DEVICES="" "$PY" -W ignore - <<'EOF'
import time
import numpy as np
import torch
assert not torch.cuda.is_available()
import mmcv, mmdet, mmengine
from mmdet.apis import init_detector, inference_detector
t0 = time.time()
model = init_detector(
    "mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_finebio.py",
    "checkpoints/dino.pth",
    device="cpu",
)
load = time.time() - t0
t1 = time.time()
result = inference_detector(model, np.zeros((800, 1333, 3), dtype=np.uint8))
infer = time.time() - t1
inst = result.pred_instances
print(
    f"[install_finebio_detector] ok: torch={torch.__version__} mmcv={mmcv.__version__} "
    f"mmengine={mmengine.__version__} mmdet={mmdet.__version__} "
    f"classes={len(model.dataset_meta['classes'])} num_classes={model.bbox_head.num_classes} "
    f"queries={len(inst.scores)} max_score={float(inst.scores.max()):.3f} "
    f"load={load:.1f}s infer={infer:.1f}s"
)
EOF
fi
log "done ($FINEBIO_DET_DIR; mmdetection @ $(git -C mmdetection rev-parse --short HEAD))"
