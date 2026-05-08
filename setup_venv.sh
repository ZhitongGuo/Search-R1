#!/bin/bash
set -eo pipefail

# =============================================================================
# Search-R1 Setup — venv-based (A100 compatible)
# All conda envs on this machine are sm_90a only. Using pip venv instead.
# =============================================================================

SEARCH_R1_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SEARCH_R1_DIR/.venv"

export https_proxy="${https_proxy:-http://fwdproxy:8080}"
export http_proxy="${http_proxy:-http://fwdproxy:8080}"
export HF_HUB_DISABLE_XET=1

echo "============================================="
echo "Search-R1 Setup — venv (A100 sm_80)"
echo "============================================="

# -------------------------------------------------------------------------
# Step 1: Create venv + install torch with A100 support
# -------------------------------------------------------------------------
echo ""
echo "=== Step 1/3: Creating venv + installing packages ==="

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/python3" ]; then
    echo "  Venv already exists."
else
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# torch with CUDA 12.8 (supports sm_80 A100)
pip install --proxy http://fwdproxy:8080 torch --index-url https://download.pytorch.org/whl/cu128

# vllm (latest stable, supports A100)
pip install --proxy http://fwdproxy:8080 vllm

# verl
pip install --proxy http://fwdproxy:8080 verl

# Other deps
pip install --proxy http://fwdproxy:8080 \
    transformers datasets accelerate \
    hydra-core omegaconf codetiming \
    ray wandb \
    faiss-cpu uvicorn fastapi \
    huggingface_hub

# flash-attn is optional (needs matching CUDA toolkit to compile)
pip install --proxy http://fwdproxy:8080 flash-attn --no-build-isolation 2>/dev/null || \
    echo "  WARNING: flash-attn failed to build (optional, training works without it)"

echo "  torch:  $(python3 -c 'import torch; print(torch.__version__, "archs:", torch.cuda.get_arch_list())')"
echo "  vllm:   $(python3 -c 'from importlib.metadata import version; print(version("vllm"))')"
echo "  verl:   $(python3 -c 'from importlib.metadata import version; print(version("verl"))')"

# -------------------------------------------------------------------------
# Step 2: Rename bundled verl/ if not already done
# -------------------------------------------------------------------------
echo ""
echo "=== Step 2/3: Handling bundled verl fork ==="
if [ -d "$SEARCH_R1_DIR/verl" ] && [ ! -d "$SEARCH_R1_DIR/_verl_bundled" ]; then
    mv "$SEARCH_R1_DIR/verl" "$SEARCH_R1_DIR/_verl_bundled"
    echo "  Renamed verl/ -> _verl_bundled/"
else
    echo "  Already handled."
fi

# -------------------------------------------------------------------------
# Step 3: Process datasets (if not already done)
# -------------------------------------------------------------------------
echo ""
echo "=== Step 3/3: Processing datasets ==="
cd "$SEARCH_R1_DIR"

DATA_DIR="$SEARCH_R1_DIR/data"

if [ -f "$DATA_DIR/nq_search/train.parquet" ]; then
    echo "  NQ: already processed."
else
    echo "  Processing NQ..."
    mkdir -p "$DATA_DIR/nq_search"
    python3 scripts/data_process/nq_search.py --local_dir "$DATA_DIR/nq_search"
fi

echo ""
echo "============================================="
echo "Setup complete!"
echo "============================================="
echo ""
echo "Activate with: source $VENV_DIR/bin/activate"
echo ""
echo "Launch:"
echo "  1. bash launch_retriever.sh"
echo "  2. bash train_grpo_3b.sh"
