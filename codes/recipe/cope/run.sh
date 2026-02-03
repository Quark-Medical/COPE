# Set processed files for training
DATASET_FILES="['../data/train.parquet']"
TEST_DATASET_FILES="['../data/test.parquet']"

# Setup tensorboard logging
export TENSORBOARD_DIR="../logs/tensorboard"

export SAVE_CHECKPOINT_DIR='../save_checkpoints'
export OUTPUT_DIR='../outputs'

# Setup actor checkpoint
export ACTOR_LOAD="../model/Qwen/Qwen3-1.7B"

# Resource info
export NNODES=${NNODES}
export NODE_RANK=${RANK}

# ================= algorithm =================
adv_estimator=gae

use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=True
kl_loss_coef=0.001

clip_ratio_low=0.2
clip_ratio_high=0.28

max_turns=1 
max_prompt_length=512
max_response_length=32000
rm_max_prompt_len=1792
rm_max_response_len=1280
assistant_max_prompt_len=640
assistant_max_response_len=1280

actor_lr=1e-6

train_batch_size=256
ppo_mini_batch_size=256
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
n_resp_per_prompt=1
n_resp_per_prompt_val=1

# ================= perfomance =================
infer_tp=2 # vllm rollout tp
train_sp=1 # train
offload=True

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 1 ))

python3 -m recipe.cope.mt_main \
        algorithm.adv_estimator=${adv_estimator} \
        algorithm.use_kl_in_reward=$use_kl_in_reward \
        algorithm.kl_ctrl.kl_coef=$kl_coef \
        algorithm.gamma=1.0 \
        algorithm.lam=1.0 \
        data.train_files="$DATASET_FILES" \
        data.val_files="$TEST_DATASET_FILES" \
        data.return_raw_chat=True \
        data.train_batch_size=$train_batch_size \
        data.max_prompt_length=$max_prompt_length \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.mode='single-domain' \
        data.shuffle=False \
        data.apply_chat_template_kwargs.enable_thinking=False \
        actor_rollout_ref.model.path=${ACTOR_LOAD} \
        actor_rollout_ref.model_type=qwen3 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.ppo_epochs=1 \
        actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
        actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
        actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
        actor_rollout_ref.actor.clip_ratio_c=10.0 \
        actor_rollout_ref.actor.optim.lr=$actor_lr \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
        actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
        actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.response_length=$max_response_length \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
        actor_rollout_ref.rollout.multi_turn.enable=True \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.n=$n_resp_per_prompt \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.calculate_log_probs=False \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
        actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.agent.agent_loop_config_path='recipe/cope/agent.yaml' \
        actor_rollout_ref.rollout.agent.num_workers=8 \
        reward_model.reward_manager='personal_reward_manager' \
        reward_model.enable=False \
        reward_model.loop_enable=False \
        critic.model.path=${ACTOR_LOAD} \
        critic.model.use_remove_padding=True \
        critic.model.fsdp_config.param_offload=False \
        critic.model.fsdp_config.optimizer_offload=False \
        critic.optim.lr=1e-5 \
        critic.enable=True \
        rm_critic.model.path=${ACTOR_LOAD} \
        rm_critic.model.use_remove_padding=True \
        rm_critic.model.fsdp_config.param_offload=False \
        rm_critic.model.fsdp_config.optimizer_offload=False \
        rm_critic.optim.lr=1e-5 \
        rm_critic.enable=True \
        trainer.logger=['console','tensorboard'] \
        trainer.project_name='multi-turn-rl' \
        trainer.experiment_name='rl' \
        trainer.n_gpus_per_node=8 \
        trainer.critic_warmup=5 \
        trainer.val_before_train=False \
        trainer.log_val_generations=0 \
        trainer.nnodes=${NNODES} \
        trainer.save_freq=5 \
        trainer.default_local_dir=${SAVE_CHECKPOINT_DIR} \
        trainer.rollout_data_dir=${OUTPUT_DIR}/rollout_data \
        trainer.validation_data_dir=${OUTPUT_DIR}/validation_data \
        trainer.test_freq=-1 \
        trainer.total_epochs=1 \
        trainer.balance_batch=False \
        personal.use_personalization_vector=True \
        personal.use_sft_loss=True \
        personal.use_rm_loss=True \
        personal.include_past_history=False \
        personal.include_past_history_len=200 \
        personal.real_reward_usage_prob=0.75 \
        personal.real_reward_user_note_prob=0.25 \
        personal.user_prefix_len=10 \
        +personal.num_train_tasks=51 \
        +personal.reset_step_ids='[68,85]' \
        personal.user_states_save_path=${OUTPUT_DIR}/user_states \
        personal.save_all_user_states_freq=5 \
        personal.user.async_request_mode='bailian' \
        personal.rollout.rm_max_prompt_len=$rm_max_prompt_len \
        personal.rollout.rm_max_response_len=$rm_max_response_len \
        personal.rollout.assistant_max_prompt_len=$assistant_max_prompt_len \
        personal.rollout.assistant_max_response_len=$assistant_max_response_len \
        actor_rollout_ref.p_vec_lr=1e-3 \
        trainer.val_only=False \
