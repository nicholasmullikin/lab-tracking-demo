#!/usr/bin/env bash
# Idempotent install of the FineBio object detector (Yagi et al., IJCV 2025): MMDetection 3.x
# with the authors' DINO / Deformable DETR configs and released weights, in its own venvs at
# /home/nick/src/finebio-detector. Never touches the battle env.
#
# Two targets, chosen by --cuda / FINEBIO_CUDA=1 (default: cpu):
#
#   cpu  (.venv, Sep 21)       torch 2.1.2+cpu, the prebuilt OpenMMLab CPU mmcv 2.1.0 wheel.
#                              DINO's MultiScaleDeformableAttention falls back to the
#                              pure-PyTorch path off CUDA, so no CUDA extension is needed.
#                              ~2-4 s/frame at 1333x800 on 32 threads.
#   cuda (.venv-cuda, Sep 24)  torch 2.13.0+cu130 / torchvision 0.28.0+cu130 (the torch line
#                              the MuggledSAM env runs; sm_120 in its arch list), mmcv 2.1.0
#                              built from the PyPI sdist with its CUDA ops for sm_120 only.
#                              ~50-55 ms/frame, < 1 GiB VRAM. Record of the Sep 24 build:
#                                - no source patch was needed: torch 2.13 still ships
#                                  THC/THCAtomics.cuh and c10::optional (= std::optional),
#                                  the only two legacy APIs mmcv 2.1.0's ops use;
#                                - CUDA 13.2's nvcc refuses GCC 16 as host compiler
#                                  ("unsupported GNU version"), so the build passes
#                                  CC=gcc-15 CXX=g++-15 (torch's cpp_extension turns $CC into
#                                  nvcc -ccbin); FINEBIO_HOST_CC / FINEBIO_HOST_CXX override;
#                                - the compile took 2.5 min with MAX_JOBS=16 on 32 cores;
#                                - torch >= 2.6 loads checkpoints weights-only by default and
#                                  the authors' .pth carry mmengine HistoryBuffers, so the
#                                  worker (battle.finebio_detect) sets
#                                  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 for its own process;
#                                - cuDNN's TF32 convolutions move scores by up to ~0.03 vs
#                                  the CPU reference; the worker disables TF32 by default
#                                  (--tf32 to allow), then boxes agree to 0.2 px / 0.009.
#   Both venvs share the mmdetection checkout (editable install into each), the FineBio
#   configs and the checkpoints; the shared steps are skipped when already done.
#
# Steps (each skipped when already done):
#   1. uv venv (Python 3.10)
#   2. torch + torchvision (cpu: PyTorch CPU index; cuda: cu130 index)
#   3. mmengine 0.10.7, numpy<2 (mmcv 2.1.0 is compiled against numpy 1.x)
#   4. mmcv 2.1.0 with ops (cpu: prebuilt CPU wheel, sdist fallback; cuda: sdist build)
#   5. mmdetection cloned at v3.3.0 (the last release accepting mmcv<2.2.0), editable install
#   6. the FineBio object_detection files fetched at a pinned aistairc/FineBio commit
#      (two configs + their README), copied into mmdetection/configs/{dino,deformable_detr}/
#      as the authors' README instructs
#   7. the released weights from the authors' Google Drive links (the README's 2025-12-15
#      update; the older finebio.s3.abci.ai host no longer resolves) via gdown, sha256 recorded
#      in checkpoints/SHA256SUMS
#   8. a verification: init_detector + inference_detector on a blank image (cpu: CUDA hidden;
#      cuda: on cuda:0, skipped while a Battle GPU worker is on the card)
#
# Usage: scripts/install_finebio_detector.sh [--cuda] [--skip-verify] [--skip-weights]
set -euo pipefail

FINEBIO_DET_DIR="${FINEBIO_DET_DIR:-/home/nick/src/finebio-detector}"
MMDET_COMMIT="${MMDET_COMMIT:-44ebd17b145c2372c4b700bfb9cb20dbd28ab64a}" # tag v3.3.0
FINEBIO_COMMIT="${FINEBIO_COMMIT:-cb8d16ef13c7c9901418c13bcbaa50a3bdf3a2c3}" # aistairc/FineBio main, 2026-09-21
MMCV_VERSION=2.1.0
MMENGINE_VERSION=0.10.7
# cpu target
CPU_TORCH_VERSION=2.1.2
CPU_TORCHVISION_VERSION=0.16.2
MMCV_WHEEL_URL="https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/mmcv-${MMCV_VERSION}-cp310-cp310-manylinux1_x86_64.whl"
# cuda target
CUDA_TORCH_VERSION=2.13.0
CUDA_TORCHVISION_VERSION=0.28.0
CUDA_TORCH_INDEX="https://download.pytorch.org/whl/cu130"
MMCV_SDIST_URL="https://files.pythonhosted.org/packages/source/m/mmcv/mmcv-${MMCV_VERSION}.tar.gz"
MMCV_SDIST_SHA256="d387bcab66b467479b6660310e23746cfc79c6e57acf04094680adb499a5cd3f"
TORCH_CUDA_ARCH_LIST_DEFAULT="12.0" # RTX 5070 Ti (Blackwell, sm_120) only
FINEBIO_HOST_CC="${FINEBIO_HOST_CC:-gcc-15}"
FINEBIO_HOST_CXX="${FINEBIO_HOST_CXX:-g++-15}"
CUDA_HOME_DEFAULT="${CUDA_HOME:-/usr/local/cuda}"
FINEBIO_RAW="https://raw.githubusercontent.com/aistairc/FineBio/${FINEBIO_COMMIT}/object_detection"
DINO_GDRIVE_ID="1VseplD0tPLJ89mzD5IGbCa16_gooNqXy"
DDETR_GDRIVE_ID="1U390LjTByULtgD6XlecWrHPAlU9AjH_W"
TARGET="cpu"
[ "${FINEBIO_CUDA:-0}" = "1" ] && TARGET="cuda"
SKIP_VERIFY=0
SKIP_WEIGHTS=0
for arg in "$@"; do
  case "$arg" in
    --cuda) TARGET="cuda" ;;
    --cpu) TARGET="cpu" ;;
    --skip-verify) SKIP_VERIFY=1 ;;
    --skip-weights) SKIP_WEIGHTS=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '[install_finebio_detector] %s\n' "$*"; }

mkdir -p "$FINEBIO_DET_DIR"/{checkpoints,finebio_configs}
cd "$FINEBIO_DET_DIR"

if [ "$TARGET" = "cuda" ]; then
  VENV=".venv-cuda"
else
  VENV=".venv"
fi
if [ ! -x "$VENV/bin/python" ]; then
  log "creating $VENV (python 3.10)"
  uv venv --python 3.10 "$VENV"
fi
PY="$FINEBIO_DET_DIR/$VENV/bin/python"

# ---- 2. torch -------------------------------------------------------------------------------
if [ "$TARGET" = "cuda" ]; then
  if ! "$PY" -c "import torch, torchvision; assert torch.__version__ == '${CUDA_TORCH_VERSION}+cu130' and 'sm_120' in torch.cuda.get_arch_list()" 2>/dev/null; then
    log "installing torch ${CUDA_TORCH_VERSION}+cu130 / torchvision ${CUDA_TORCHVISION_VERSION}+cu130"
    uv pip install --python "$PY" "torch==${CUDA_TORCH_VERSION}" "torchvision==${CUDA_TORCHVISION_VERSION}" \
      --index-url "$CUDA_TORCH_INDEX"
  fi
else
  if ! "$PY" -c "import torch, torchvision; assert torch.__version__.startswith('${CPU_TORCH_VERSION}+cpu')" 2>/dev/null; then
    log "installing torch ${CPU_TORCH_VERSION}+cpu / torchvision ${CPU_TORCHVISION_VERSION}+cpu"
    uv pip install --python "$PY" "torch==${CPU_TORCH_VERSION}" "torchvision==${CPU_TORCHVISION_VERSION}" \
      --index-url https://download.pytorch.org/whl/cpu
  fi
fi

# ---- 3. mmengine, numpy<2 ----------------------------------------------------------------------
if ! "$PY" -c "import mmengine, numpy; assert mmengine.__version__ == '${MMENGINE_VERSION}' and numpy.__version__ < '2'" 2>/dev/null; then
  log "installing mmengine ${MMENGINE_VERSION}, numpy<2"
  uv pip install --python "$PY" "mmengine==${MMENGINE_VERSION}" "numpy<2"
fi

# ---- 4. mmcv with ops ----------------------------------------------------------------------------
mmcv_ok() {
  "$PY" - <<'EOF' 2>/dev/null
import mmcv
from mmcv.ops import multi_scale_deform_attn, nms  # both need the compiled _ext
assert mmcv.__version__ == "2.1.0", mmcv.__version__
EOF
}
mmcv_cuda_ok() {
  mmcv_ok && "$PY" - <<'EOF' 2>/dev/null
import mmcv.ops
assert mmcv.ops.get_compiling_cuda_version() not in (None, "n/a", ""), "mmcv ops compiled without CUDA"
EOF
}
if [ "$TARGET" = "cuda" ]; then
  if ! mmcv_cuda_ok; then
    log "building mmcv ${MMCV_VERSION} from the PyPI sdist with CUDA ops (arch ${TORCH_CUDA_ARCH_LIST:-$TORCH_CUDA_ARCH_LIST_DEFAULT})"
    sdist="$FINEBIO_DET_DIR/mmcv-${MMCV_VERSION}.tar.gz"
    if [ ! -s "$sdist" ]; then
      curl -fsSL --retry 3 -o "$sdist.part" "$MMCV_SDIST_URL" && mv "$sdist.part" "$sdist"
    fi
    echo "${MMCV_SDIST_SHA256}  ${sdist}" | sha256sum -c - >/dev/null
    [ -d "mmcv-${MMCV_VERSION}" ] || tar xzf "$sdist"
    for tool in "$FINEBIO_HOST_CC" "$FINEBIO_HOST_CXX" "$CUDA_HOME_DEFAULT/bin/nvcc"; do
      command -v "$tool" >/dev/null 2>&1 || { log "missing $tool (CUDA 13.2 needs a GCC <= 15 host compiler); stopping"; exit 1; }
    done
    uv pip uninstall --python "$PY" mmcv >/dev/null 2>&1 || true
    uv pip install --python "$PY" "setuptools<80" wheel ninja psutil
    # The build/ directory inside the sdist tree keeps ninja's objects, so a re-run after a
    # failure resumes rather than restarts.
    CC="$FINEBIO_HOST_CC" CXX="$FINEBIO_HOST_CXX" CUDA_HOME="$CUDA_HOME_DEFAULT" \
      MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$TORCH_CUDA_ARCH_LIST_DEFAULT}" \
      MAX_JOBS="${MAX_JOBS:-16}" \
      uv pip install --python "$PY" --no-build-isolation "./mmcv-${MMCV_VERSION}" "numpy<2"
    mmcv_cuda_ok || { log "mmcv ${MMCV_VERSION} does not import with CUDA ops; stopping"; exit 1; }
  fi
else
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
fi

# ---- 5. mmdetection (shared checkout, editable into this venv) -------------------------------
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
  log "installing mmdetection (editable) + inference deps into $VENV"
  # mmdet's setup.py imports torch.utils.cpp_extension, which on torch 2.1 still imports
  # pkg_resources (removed in setuptools 80), so the build cannot be isolated from the venv.
  uv pip install --python "$PY" "setuptools<80" wheel
  uv pip install --python "$PY" --no-build-isolation -e ./mmdetection "numpy<2"
fi

# ---- 6. FineBio configs -------------------------------------------------------------------------
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

# ---- 7. weights -------------------------------------------------------------------------------
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

# ---- 8. verification ----------------------------------------------------------------------------
if [ "$SKIP_VERIFY" -eq 0 ]; then
  if [ "$TARGET" = "cuda" ]; then
    if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read -r pid; do
         tr '\0' ' ' < "/proc/${pid%%,*}/cmdline" 2>/dev/null; echo; done | grep -qE "muggled_worker|four_part_video_worker|dam4sam_video_worker|finebio_detect"; then
      log "a Battle GPU worker is on the card; skipping the GPU verification (re-run later)"
    else
      log "GPU verification: DINO config + weights on cuda:0, one blank 1333x800 image"
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "$PY" -W ignore - <<'EOF'
import time
import numpy as np
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
import mmcv, mmdet, mmengine
import mmcv.ops
from mmdet.apis import init_detector, inference_detector
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
t0 = time.time()
model = init_detector(
    "mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_finebio.py",
    "checkpoints/dino.pth",
    device="cuda:0",
)
load = time.time() - t0
image = np.zeros((800, 1333, 3), dtype=np.uint8)
inference_detector(model, image)  # warm-up (kernel load) is not the number to report
torch.cuda.synchronize()
t1 = time.time()
result = inference_detector(model, image)
torch.cuda.synchronize()
infer = time.time() - t1
inst = result.pred_instances
print(
    f"[install_finebio_detector] ok: torch={torch.__version__} mmcv={mmcv.__version__} "
    f"(cuda {mmcv.ops.get_compiling_cuda_version()}, {mmcv.ops.get_compiler_version()}) "
    f"mmengine={mmengine.__version__} mmdet={mmdet.__version__} "
    f"device={torch.cuda.get_device_name(0)} arch={torch.cuda.get_arch_list()} "
    f"classes={len(model.dataset_meta['classes'])} queries={len(inst.scores)} "
    f"max_score={float(inst.scores.max()):.3f} load={load:.1f}s infer={1000 * infer:.0f}ms "
    f"peak_reserved={torch.cuda.max_memory_reserved() // 2**20}MiB"
)
EOF
    fi
  else
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
fi
log "done ($FINEBIO_DET_DIR/$VENV, target $TARGET; mmdetection @ $(git -C mmdetection rev-parse --short HEAD))"
