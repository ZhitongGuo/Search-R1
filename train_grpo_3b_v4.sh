#!/bin/bash
set -eo pipefail

# =============================================================================
# Search-R1 GRPO — Qwen2.5-3B v4 (FROM SCRATCH)
# Original v1 hyperparams (kl_coef=0.001, entropy_coeff=0.001)
# + search_r1 parser + stop strings (actual retrieval)
# + flash_attn optimizations
# =============================================================================

cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_PRELOAD="/usr/local/cuda/lib64/libcublas.so.12:/usr/local/cuda/lib64/libcublasLt.so.12"
export RAY_DISABLE_DASHBOARD=1
export VLLM_USE_V1=1

EXPERIMENT_NAME="nq-grpo-qwen2.5-3b-v4-$(date +%m%d)"

echo "============================================="
echo "v4: From scratch, original hyperparams + search fix"
echo "kl_coef=0.001, entropy_coeff=0.001"
echo "============================================="

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=data/nq_search/train.parquet \
    data.val_files=data/nq_search/test.parquet \
    data.train_batch_size=256 \
    data.val_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=models/Qwen2.5-3B \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    "actor_rollout_ref.rollout.multi_turn.tool_config_path=$(pwd)/tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=500 \
    actor_rollout_ref.rollout.multi_turn.format=search_r1 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    "reward.custom_reward_function.path=$(pwd)/reward_fn.py" \
    reward.custom_reward_function.name=compute_score \
    "trainer.logger=['console']" \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.resume_mode=disable \
    trainer.project_name=Search-R1 \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    2>&1 | tee "logs/${EXPERIMENT_NAME}.log"
