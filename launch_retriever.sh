#!/bin/bash
set -euo pipefail

# Launch the E5 dense retriever server on port 8000
# Run this in a separate tmux pane: conda activate retriever && bash launch_retriever.sh

SEARCH_R1_DIR="$(cd "$(dirname "$0")" && pwd)"
CORPUS_DIR="$SEARCH_R1_DIR/corpus"

INDEX_FILE="$CORPUS_DIR/e5_Flat.index"
CORPUS_FILE="$CORPUS_DIR/wiki-18.jsonl"
RETRIEVER_NAME="e5"
RETRIEVER_MODEL="$SEARCH_R1_DIR/models/e5-base-v2"

if [ ! -d "$RETRIEVER_MODEL" ]; then
    RETRIEVER_MODEL="intfloat/e5-base-v2"
fi

for f in "$INDEX_FILE" "$CORPUS_FILE"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found. Run setup.sh first."
        exit 1
    fi
done

echo "Launching retriever: $RETRIEVER_NAME on port 8000"

# Put retriever on GPU 7 to avoid competing with training rollouts
CUDA_VISIBLE_DEVICES=7 python3 "$SEARCH_R1_DIR/search_r1/search/retrieval_server.py" \
    --index_path "$INDEX_FILE" \
    --corpus_path "$CORPUS_FILE" \
    --topk 3 \
    --retriever_name "$RETRIEVER_NAME" \
    --retriever_model "$RETRIEVER_MODEL" \
    --faiss_gpu
