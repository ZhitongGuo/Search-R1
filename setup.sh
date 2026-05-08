#!/bin/bash
set -eo pipefail

# =============================================================================
# Search-R1 Baseline Setup — Route A (EMNLP 2026)
# Uses existing conda env: xlformers_msl_rl_conda-newest
#   (torch 2.12, vllm 0.11.1, ray 2.48, transformers 4.57)
# + upstream verl (pip install)
#
# Run from your terminal (needs proxy for PyPI/HuggingFace)
# =============================================================================

SEARCH_R1_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SEARCH_R1_DIR/data"
CORPUS_DIR="$SEARCH_R1_DIR/corpus"
MODEL_DIR="$SEARCH_R1_DIR/models"
CONDA_ENV="xlformers_msl_rl_conda-newest"

echo "============================================="
echo "Search-R1 Setup — Route A Baseline"
echo "Env: $CONDA_ENV (existing, 8× A100 80GB)"
echo "============================================="

# -------------------------------------------------------------------------
# Step 0: Activate env
# -------------------------------------------------------------------------
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate "$CONDA_ENV"
echo "Python: $(python3 --version)"
echo "Torch:  $(python3 -c 'import torch; print(torch.__version__)')"
echo "vLLM:   $(python3 -c 'from importlib.metadata import version; print(version("vllm"))')"

# -------------------------------------------------------------------------
# Step 1: Install upstream verl + small deps
# -------------------------------------------------------------------------
echo ""
echo "=== Step 1/5: Installing upstream verl ==="

# Proxy needed for PyPI/GitHub access
export https_proxy="${https_proxy:-http://fwdproxy:8080}"
export http_proxy="${http_proxy:-http://fwdproxy:8080}"
# Disable HuggingFace xet download backend (can't resolve through proxy)
export HF_HUB_DISABLE_XET=1

pip install verl codetiming 2>/dev/null || {
    echo "pip install verl from PyPI failed, trying from git..."
    pip install "git+https://github.com/volcengine/verl.git" codetiming
}

# Try to get faiss-gpu (optional, falls back to faiss-cpu which is already installed)
pip install faiss-gpu 2>/dev/null || echo "  faiss-gpu not available, using faiss-cpu (slower but works)"

echo "  verl: $(python3 -c 'from importlib.metadata import version; print(version("verl"))' 2>/dev/null || echo 'FAILED')"

# -------------------------------------------------------------------------
# Step 2: Rename bundled verl/ to avoid import shadowing
# -------------------------------------------------------------------------
echo ""
echo "=== Step 2/5: Handling bundled verl fork ==="
if [ -d "$SEARCH_R1_DIR/verl" ] && [ ! -d "$SEARCH_R1_DIR/_verl_bundled" ]; then
    echo "  Renaming verl/ -> _verl_bundled/"
    mv "$SEARCH_R1_DIR/verl" "$SEARCH_R1_DIR/_verl_bundled"
    echo "  Done. (Original code preserved in _verl_bundled/)"
elif [ -d "$SEARCH_R1_DIR/_verl_bundled" ]; then
    echo "  Already renamed."
else
    echo "  No bundled verl/ found."
fi

# -------------------------------------------------------------------------
# Step 3: Download corpus + FAISS index
# -------------------------------------------------------------------------
echo ""
echo "=== Step 3/5: Downloading corpus + index ==="
mkdir -p "$CORPUS_DIR"

if [ -f "$CORPUS_DIR/e5_Flat.index" ] && [ -f "$CORPUS_DIR/wiki-18.jsonl" ]; then
    echo "  Already downloaded."
else
    python3 "$SEARCH_R1_DIR/scripts/download.py" --save_path "$CORPUS_DIR"

    if [ -f "$CORPUS_DIR/part_aa" ] && [ -f "$CORPUS_DIR/part_ab" ]; then
        echo "  Concatenating index parts..."
        cat "$CORPUS_DIR/part_aa" "$CORPUS_DIR/part_ab" > "$CORPUS_DIR/e5_Flat.index"
        rm -f "$CORPUS_DIR/part_aa" "$CORPUS_DIR/part_ab"
    fi

    if [ -f "$CORPUS_DIR/wiki-18.jsonl.gz" ]; then
        echo "  Decompressing corpus..."
        gzip -d "$CORPUS_DIR/wiki-18.jsonl.gz"
    fi
fi

echo "  Index:  $(ls -lh "$CORPUS_DIR/e5_Flat.index" 2>/dev/null | awk '{print $5}' || echo 'missing')"
echo "  Corpus: $(ls -lh "$CORPUS_DIR/wiki-18.jsonl" 2>/dev/null | awk '{print $5}' || echo 'missing')"

# -------------------------------------------------------------------------
# Step 4: Download models
# -------------------------------------------------------------------------
echo ""
echo "=== Step 4/5: Downloading models ==="
mkdir -p "$MODEL_DIR"

for model in "Qwen/Qwen2.5-3B" "Qwen/Qwen2.5-7B" "intfloat/e5-base-v2"; do
    local_name="${model#*/}"
    if [ -d "$MODEL_DIR/$local_name" ] && [ -f "$MODEL_DIR/$local_name/config.json" ]; then
        echo "  $local_name: already downloaded."
    else
        echo "  Downloading $model..."
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$model', local_dir='$MODEL_DIR/$local_name')
"
    fi
done

# -------------------------------------------------------------------------
# Step 5: Process datasets
# -------------------------------------------------------------------------
echo ""
echo "=== Step 5/5: Processing datasets ==="
cd "$SEARCH_R1_DIR"

# NQ search (primary)
if [ -f "$DATA_DIR/nq_search/train.parquet" ]; then
    echo "  NQ: already processed."
else
    echo "  Processing NQ..."
    mkdir -p "$DATA_DIR/nq_search"
    python3 scripts/data_process/nq_search.py --local_dir "$DATA_DIR/nq_search"
fi

# Multi-dataset train
if [ -f "$DATA_DIR/multi_search/train.parquet" ]; then
    echo "  Multi-dataset: already processed."
else
    echo "  Processing multi-dataset (nq,hotpotqa,triviaqa)..."
    mkdir -p "$DATA_DIR/multi_search"
    python3 scripts/data_process/qa_search_train_merge.py \
        --local_dir "$DATA_DIR/multi_search" \
        --data_sources "nq,hotpotqa,triviaqa"
fi

# Test sets (only process what's available in cache)
for ds in nq triviaqa popqa hotpotqa; do
    if [ -f "$DATA_DIR/${ds}_search_test/test.parquet" ]; then
        echo "  Test '$ds': done."
    else
        echo "  Processing test: $ds..."
        mkdir -p "$DATA_DIR/${ds}_search_test"
        python3 scripts/data_process/qa_search_test_merge.py \
            --local_dir "$DATA_DIR/${ds}_search_test" \
            --data_sources "$ds"
    fi
done

echo ""
echo "============================================="
echo "Setup complete!"
echo "============================================="
echo ""
echo "Next steps (both use the same env: $CONDA_ENV):"
echo "  1. Tmux pane 1:  conda activate $CONDA_ENV && bash launch_retriever.sh"
echo "  2. Tmux pane 2:  conda activate $CONDA_ENV && bash train_grpo_3b.sh"
