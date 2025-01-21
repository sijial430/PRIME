#!/bin/bash
#SBATCH --job-name=test-prime
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=23:55:00
#SBATCH --mail-user=sl2998@princeton.edu
#SBATCH --partition=pli-c
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -x
module purge
module load anaconda3/2024.6
source $(conda info --base)/etc/profile.d/conda.sh
conda activate verl
wandb offline

export NCCL_DEBUG=WARN
export WANDB_API_KEY='f9c60579101e39eecec690c903bf1be978126942'
# export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1

PROJECT_NAME='PRIME'
EXPERIMENT_NAME='online-qwen-3b-after-solvable-0.2-0.8-policy-self-ref'
DATA_PATH=/home/sl2998/workspace/PRIME/dataset/prime
SFT_MODEL_PATH=Qwen/Qwen2.5-3B-Instruct  # PRIME-RL/Eurus-2-7B-SFT
CKPT_PATH=/scratch/gpfs/sl2998/models

python3 -m verl.trainer.main_ppo \
    data.train_files=["$DATA_PATH/train.parquet"] \
    data.val_files=["$DATA_PATH/validation.parquet"] \
    data.train_batch_size=256 \
    data.val_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=16 \
    trainer.test_freq=16 \
    trainer.total_epochs=1 \
    ++data.n_samples=4 \
    ++data.filter_accuracy=True \
    ++data.accuracy_lower_bound=0.2 \
    ++data.accuracy_upper_bound=0.8 \
    ++algorithm.adv_estimator=rloo \
    ++algorithm.adv_params.verifier_gamma=1.0 \
    ++algorithm.adv_params.reward_model_gamma=1.0 \
    ++reward_model.rm_type=prime \
    ++reward_model.rm_coef=5 \
    ++reward_model.prime_model.path=$SFT_MODEL_PATH  \
    ++reward_model.prime_model.ref_path=$SFT_MODEL_PATH  \
    ++reward_model.model.input_tokenizer=null \
    ++reward_model.prime_model.use_remove_padding=True \
    ++reward_model.prime_granularity=token \
    ++reward_model.micro_batch_size=8 \
    ++reward_model.prime_model.update=after \
    ++reward_model.prime_model.beta_train=0.05 \
    ++reward_model.prime_model.optim.lr=1e-6 \
    ++reward_model.prime_model.optim.grad_clip=10.0 \
    ++reward_model.prime_model.input_tokenizer=null \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \

set +x