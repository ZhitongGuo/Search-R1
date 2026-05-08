#!/bin/bash
set -eo pipefail

# =============================================================================
# Search-R1 GRPO Training — Qwen2.5-3B (fast iteration)
# 8× A100 80GB — upstream verl 0.7.1 with multi-turn search
# =============================================================================

cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ATTENTION_BACKEND=XFORMERS

# Pre-load cublas to fix vllm symbol resolution
export LD_PRELOAD="/usr/local/cuda/lib64/libcublas.so.12:/usr/local/cuda/lib64/libcublasLt.so.12${LD_PRELOAD:+:$LD_PRELOAD}"
# Disable Ray dashboard (opentelemetry incompatibility)
export RAY_DISABLE_DASHBOARD=1

WANDB_PROJECT='Search-R1'
BASE_MODEL='models/Qwen2.5-3B'
EXPERIMENT_NAME="nq-grpo-qwen2.5-3b-$(date +%m%d)"
TRAIN_DATA='data/nq_search/train.parquet'
TEST_DATA='data/nq_search/test.parquet'
REWARD_FN="$(pwd)/reward_fn.py"
TOOL_CONFIG="$(pwd)/tool_config.json"

echo "============================================="
echo "Training: $EXPERIMENT_NAME"
echo "Model:    $BASE_MODEL"
echo "============================================="

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$TEST_DATA \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.shuffle=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=128 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=256 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=500 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=256 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    reward.custom_reward_function.path=$REWARD_FN \
    reward.custom_reward_function.name=compute_score \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    2>&1 | tee "logs/${EXPERIMENT_NAME}.log"
