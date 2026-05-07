# Search-R1 Baseline — Route A Run Guide

Hardware: 8× A100 80GB
Env: `xlformers_msl_rl_conda-newest` (torch 2.12, vllm 0.11.1) + upstream verl
Target: EMNLP 2026 (ARR 2026-05-25)

## Step 1: Run Setup

From your terminal (needs proxy):
```bash
cd ~/Repos/Search-R1
bash setup.sh
```

Installs upstream verl, renames bundled `verl/` to `_verl_bundled/`, downloads models/corpus, processes datasets.

## Step 2: Launch Retriever (tmux pane 1)

```bash
conda activate xlformers_msl_rl_conda-newest
cd ~/Repos/Search-R1
bash launch_retriever.sh
```

Wait for "Uvicorn running on http://0.0.0.0:8000".

## Step 3: Launch Training (tmux pane 2)

```bash
conda activate xlformers_msl_rl_conda-newest
cd ~/Repos/Search-R1
bash train_grpo_3b.sh   # fast iteration
bash train_grpo_7b.sh   # main baseline
```

## Key Files

| File | Purpose |
|------|---------|
| `setup.sh` | One-time setup (pip only, no conda create) |
| `launch_retriever.sh` | E5 retriever on port 8000 (GPU 7) |
| `train_grpo_3b.sh` | 3B fast iteration |
| `train_grpo_7b.sh` | 7B main baseline |
| `train_main.py` | Training entry (upstream verl) |
| `reward_fn.py` | Exact match reward |
| `configs/ppo_trainer.yaml` | Default hydra config |
