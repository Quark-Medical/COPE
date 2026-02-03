# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import re
import uuid
import math
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
import copy
import torch.optim as optim
from typing import Dict, List, Any, Tuple
from tensordict import TensorDict
from omegaconf import OmegaConf, open_dict
from torch import Tensor
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from recipe.cope import core_algos
from recipe.cope.core_algos import AdvantageEstimator, agg_loss

from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = type[Worker]

from recipe.cope.algo.algos import compute_grpo_outcome_advantage
from recipe.cope.mt_metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)

import random
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from verl.utils.py_functional import convert_to_regular_types

def replace_nan(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    elif isinstance(obj, dict):
        return {k: replace_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return replace_nan(obj.tolist())
    elif isinstance(obj, (np.integer, np.floating)):
        if isinstance(obj, np.floating) and math.isnan(float(obj)):
            return None
        return obj.item()
    elif hasattr(obj, 'tolist'):
        # Handle other array-like objects that have tolist method
        return replace_nan(obj.tolist())
    elif not isinstance(obj, (str, int, float, bool, type(None))):
        # Convert other non-serializable types to string
        return str(obj)
    return obj

@dataclass
class UserState:
    user_id: str
    reset_len: int
    p_vec: torch.Tensor = None
    history: List[Dict] = field(default_factory=list)
    feedback_log: List[Dict[str, Any]] = field(default_factory=list)

class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5 
    ActorRolloutRef = 6
    RMCritic = 7 


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            adv_compute_info=data.non_tensor_batch['adv_compute_info']
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.use_p_vec = self.config.personal.get("use_personalization_vector", True)
        self.include_past_history = self.config.personal.get("include_past_history", True)
        self.include_past_history_len = self.config.personal.get("include_past_history_len", 5)
        self.real_reward_usage_prob = self.config.personal.get("real_reward_usage_prob", 0.5)

        self.all_user_states: Dict[str, UserState] = {}
        self.max_total_tasks: int = 0
        self.num_train_tasks = self.config.personal.get("num_train_tasks", 0)
        
        self.user_states_save_path = self.config.personal.get("user_states_save_path", "./user_states")
        os.makedirs(self.user_states_save_path, exist_ok=True)

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        # Policy Critic
        self.use_policy_critic = config.critic.enable
        # RM Critic
        self.use_rm_critic = config.rm_critic.enable

        self._validate_config()
        self._create_dataloader_and_initialze_all_user_states(train_dataset, val_dataset, collate_fn, train_sampler)
 
    def _include_extra_data_for_gen_batch(self, gen_batch: DataProto, sub_task_idx: int, 
                                            user_states: Dict[str, UserState]) -> DataProto:
        batch_size = len(gen_batch)

        extra_info_arr = gen_batch.non_tensor_batch["extra_info"]

        for i in range(batch_size):
            extra_info = extra_info_arr[i]
            extra_info["sub_task_idx"] = sub_task_idx
            extra_info["data_idx"] = i

            if self.include_past_history and sub_task_idx >= self.num_train_tasks:
                prompt: List = extra_info["prompt"]
                user_id = extra_info["user_id"]
                ustate = user_states[user_id]
                history = ustate.history[-self.include_past_history_len:]
                if not history:
                    continue

                new_prompt = prompt + history
                extra_info["prompt"] = new_prompt

            extra_info_arr[i] = extra_info

        gen_batch.non_tensor_batch["extra_info"] = extra_info_arr
        return gen_batch

    
    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch


    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            if config.actor_rollout_ref.actor.strategy == "megatron":
                model_parallel_size = (
                    config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                    * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
                )
                assert (
                    n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
                ), (
                    f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                    f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
                )
                megatron_dp = n_gpus // (
                    model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
                )
                minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
            else:
                minimal_bsz = n_gpus

            # 1. Check total batch size for data correctness
            real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
            assert real_train_batch_size % minimal_bsz == 0, (
                f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
                f"({minimal_bsz})"
            )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            """Validate mutually exclusive micro batch size configuration options.

            Ensures that users don't set both deprecated micro_batch_size and
            the new micro_batch_size_per_gpu parameters simultaneously.

            Args:
                mbs: Deprecated micro batch size parameter value.
                mbs_per_gpu: New micro batch size per GPU parameter value.
                name (str): Configuration section name for error messages.

            Raises:
                ValueError: If both parameters are set or neither is set.
            """
            settings = {
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        # Actor validation done in ActorConfig.__post_init__ and validate()
        actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
        actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        if self.config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_policy_critic:
            critic_config = omega_conf_to_dataclass(config.critic)
            critic_config.validate(n_gpus, config.data.train_batch_size)

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader_and_initialze_all_user_states(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        assert train_dataset is not None, "Train dataset is None!"
        assert val_dataset is None, "Val dataset is not None!"
        assert train_sampler is not None, "Train sampler is None!"
        assert collate_fn is not None, "Collate fn is None!"

        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=False,  
            collate_fn=collate_fn,
            sampler=train_sampler,
        )
        self._initial_dataloader_state = self.train_dataloader.state_dict()

        self.val_dataloader = None


        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}"
        )

        print("Initializing global states and calculating total steps...")

        max_total_tasks = 0
        steps_per_epoch = 0

        for batch_dict in tqdm(self.train_dataloader, desc="Initializing & Calculating"):
            # batch_dict: {key: Tensor or np.ndarray[batch]}
            extra_info_arr = batch_dict["extra_info"]          # numpy array
            user_ids = np.array([ei["user_id"] for ei in extra_info_arr], dtype=object)
            sub_tasks_arr = np.array([ei["tasks"] for ei in extra_info_arr], dtype=object)

            batch_max_tasks = 0

            for u_id, sub_tasks in zip(user_ids, sub_tasks_arr):
                user_id = str(u_id)  
                if user_id not in self.all_user_states:
                    self.all_user_states[user_id] = UserState(user_id=user_id, reset_len=0, p_vec=None)

                num_user_tasks = len(sub_tasks)
                max_total_tasks = max(max_total_tasks, num_user_tasks)
                batch_max_tasks = max(batch_max_tasks, num_user_tasks)

            steps_per_epoch += batch_max_tasks

        self.max_total_tasks = max_total_tasks
        print(f"Maximum number of sub-tasks across all users: {self.max_total_tasks}")
        
        self.total_training_steps = steps_per_epoch * self.config.trainer.total_epochs

        print(f"Calculated total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = self.config.personal.num_train_tasks
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = self.config.personal.num_train_tasks
                if OmegaConf.select(self.config, "rm_critic.optim"):
                    self.config.rm_critic.optim.total_training_steps = self.config.personal.num_train_tasks
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Error: {e}")
    def _save_all_user_states(self, sub_task_idx: int):
        states_path = os.path.join(self.user_states_save_path, f"all_user_states_{sub_task_idx}.jsonl")
        
        with open(states_path, "w", encoding="utf-8") as f:
            for user_id, state in self.all_user_states.items():
                data = {
                    "user_id": user_id,
                    "history": state.history,
                    "feedback_log": state.feedback_log,
                }
                
                if hasattr(state, 'p_vec') and state.p_vec is not None:
                    if isinstance(state.p_vec, np.ndarray):
                        data["p_vec"] = state.p_vec.tolist()
                    elif hasattr(state.p_vec, 'tolist'): 
                        data["p_vec"] = state.p_vec.tolist()
                    else:
                        data["p_vec"] = state.p_vec

                line = json.dumps(data, ensure_ascii=False)
                f.write(line + "\n")

        print(f"Successfully dumped {len(self.all_user_states)} user states to {states_path}")

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create policy critic
        if self.use_policy_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["policy_critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # critic rm critic (New)
        if self.use_rm_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RMCritic)
            rm_critic_cfg = omega_conf_to_dataclass(self.config.rm_critic)
            rm_critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.RMCritic], config=rm_critic_cfg)
            self.resource_pool_to_cls[resource_pool]["rm_critic"] = rm_critic_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        #------------------------------------------USELESS----------------------------------------
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_policy_critic:
            self.policy_critic_wg = all_wg["policy_critic"]
            self.policy_critic_wg.init_model()
        if self.use_rm_critic:
            self.rm_critic_wg = all_wg["rm_critic"]
            self.rm_critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from recipe.cope.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_policy_critic:
            policy_critic_local_path = os.path.join(local_global_step_folder, "policy_critic")
            policy_critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "policy_critic")
            )
            self.policy_critic_wg.save_checkpoint(
                policy_critic_local_path, policy_critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        if self.use_rm_critic:
            rm_critic_local_path = os.path.join(local_global_step_folder, "rm_critic")
            rm_critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "rm_critic")
            )
            self.rm_critic_wg.save_checkpoint(
                rm_critic_local_path, rm_critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _dump_generations(self, inputs, outputs, dump_path, **kwargs):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "step": [self.global_steps] * n,
        }

        # Add any additional kwargs to base_data
        for key, value in kwargs.items():
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) == n:
                # If it's a sequence with correct length, use as is
                base_data[key] = value
            elif not isinstance(value, (list, tuple, np.ndarray)):
                # If it's a scalar, replicate for all samples
                base_data[key] = [value] * n
            else:
                # Log warning for mismatched lengths but still include
                print(f"Warning: Length mismatch for key '{key}': expected {n}, got {len(value)}")
                base_data[key] = value

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            entry = replace_nan(entry)
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _get_reward_model_scores(self, personalization, assistant_personalization):
        user_mapped = torch.zeros_like(personalization)
        user_mapped[personalization >= 9] = 3
        user_mapped[(personalization >= 6) & (personalization <= 8)] = 2
        user_mapped[(personalization >= 3) & (personalization <= 5)] = 1
        user_mapped[personalization <= 2] = 0
        rm_update_rewards = (3.0 - torch.abs(user_mapped - assistant_personalization)) / 3.0 + \
                            (user_mapped == assistant_personalization).float()

        pairs = torch.stack([user_mapped, assistant_personalization], dim=1)
        unique_pairs, inverse_indices, counts = torch.unique(pairs, dim=0, return_inverse=True, return_counts=True)
        pair_frequencies = counts[inverse_indices]
        freq_weights = 1 / pair_frequencies.float()
        range_tensor = torch.arange(4, device=pairs.device)
        target_pairs = torch.cartesian_prod(range_tensor, range_tensor) 

        mask = (pairs.unsqueeze(1) == target_pairs).all(dim=-1).any(dim=-1)
        rm_update_weights = torch.ones_like(freq_weights)
        if mask.any():
            target_raw_weights = freq_weights[mask]
            max_val = target_raw_weights.max()
            rm_update_weights[mask] = 1.0 * (target_raw_weights / max_val)
        return rm_update_rewards, rm_update_weights

    def _get_rewards_and_masks(
        self, data: DataProto, sub_task_idx: int) -> Tuple[Any]:
        """
        Computes real and estimated rewards, and determines which samples to use for RM and Policy updates.
        This function handles the per-user logic for reward selection.
        """
        metrics = {}
        batch_size = len(data.non_tensor_batch['user_id'])
        device = data.batch.device

        feedback_mask_arrray = data.non_tensor_batch.get('feedback_mask')
        feedback_mask = torch.tensor(feedback_mask_arrray, dtype=torch.bool, device=device)
        if feedback_mask.sum() == 0:
            feedback_mask[0] = 1 
            print(f"[NOTEIT] feedback_mask is all False, set the first to True!!!!")

        # 9,10 -> 3 (A)
        # 6,7,8 -> 2 (B)
        # 3,4,5 -> 1 (C)
        # 0,1,2 -> 0 (D)
        personalization = torch.tensor(data.non_tensor_batch['user_personalization_score'], device=device)
        user_mapped = torch.zeros_like(personalization)
        user_mapped[personalization >= 9] = 3
        user_mapped[(personalization >= 6) & (personalization <= 8)] = 2
        user_mapped[(personalization >= 3) & (personalization <= 5)] = 1
        user_mapped[personalization <= 2] = 0
        metrics["rewards/user_personalization_mapped_mean"] = user_mapped.mean().item()
        
        # 1. Get real rewards from user agent feedback
        completeness = torch.tensor(data.non_tensor_batch['user_completeness_score'], device=device)
        real_rewards = completeness * user_mapped
        metrics['rewards/completeness_mean'] = completeness.mean().item()
        metrics['rewards/user_personalization_mean'] = personalization.mean().item()
        metrics["rewards/real_mean"] = real_rewards.mean().item()

        # 2. Get estimated rewards from the reward model (which is the assistant model)
        assistant_personalization = torch.tensor(data.non_tensor_batch['assistant_personalization_score'], device=device)
        estimated_rewards = completeness * assistant_personalization
        metrics['rewards/rm_personalization_mean'] = assistant_personalization.mean().item()
        metrics["rewards/estimated_mean"] = estimated_rewards.mean().item()

        # 3. Decide for each user which reward to use for policy update and RM update
        policy_rewards = torch.where(feedback_mask, real_rewards, estimated_rewards)

        metrics["policy/used_real_feedback_ratio"] = feedback_mask.float().mean().item()
        metrics["rewards/policy_mean"] = policy_rewards.mean().item()

        rm_update_rewards, rm_update_weights = self._get_reward_model_scores(personalization, assistant_personalization)
        metrics["rewards/rm_update_mean"] = rm_update_rewards.mean().item()
        metrics["rewards/rm_update_weights_mean"] = rm_update_weights.mean().item()
        return policy_rewards, real_rewards, estimated_rewards, completeness, personalization, \
                    assistant_personalization, feedback_mask, rm_update_rewards, metrics, rm_update_weights
    
    def _filter_batch_and_rewards_by_mask(self, batch: DataProto, mask: torch.Tensor, rewards: torch.Tensor, rm_update_weights: torch.Tensor) -> \
    tuple[DataProto, Tensor, Tensor]:
        num_active = mask.sum()
        assert num_active > 0, "No active users"
        
        batch = batch.select_idxs(idxs=mask)
        rewards = rewards[mask]
        rm_update_weights = rm_update_weights[mask]

        return batch, rewards, rm_update_weights
    
    def _get_sft_loss_mask(self, batch: DataProto) -> torch.Tensor:
        response_len = batch.batch["responses"].size(1)
        input_ids = batch.batch["input_ids"]
        response_input_ids = input_ids[:, -response_len:]
        response_mask = batch.batch["response_mask"]  

        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32, device=batch.batch.device)

        is_pad = response_input_ids == self.tokenizer.pad_token_id

        loss_mask[:, -response_len:] = (1 - response_mask) * (~is_pad)

        return loss_mask

    def _perform_sft_update(self, batch: DataProto, feedback_mask: torch.Tensor, timing_raw: dict) -> dict:

        assert feedback_mask.sum() > 0, "No active users"
        batch = batch.select_idxs(idxs=feedback_mask)

        with marked_timer("sft_update_prep", timing_raw, color="purple"):

            # Ensure response_mask exists
            assert "responses" in batch.batch

            batch.batch["loss_mask"] = self._get_sft_loss_mask(batch)

            # Set the flag for the worker to trigger SFT logic
            batch.meta_info["sft_update"] = True

        # Pad the batch to the size divisor
        size_divisor = self.actor_rollout_wg.world_size
        batch, pad_size = pad_dataproto_to_divisor(batch, size_divisor)
        
        with marked_timer("sft_update_actor", timing_raw, color="purple"):
            actor_output = self.actor_rollout_wg.update_actor(batch)
        
        # Collect and prefix metrics
        sft_metrics = reduce_metrics(actor_output.meta_info["metrics"])
        prefixed_metrics = {f"sft_update/{k}": v for k, v in sft_metrics.items()}
        
        return prefixed_metrics

    def _perform_rl_update(self, batch: DataProto, update_name: str, timing_raw: dict,
                           rewards: torch.Tensor, rm_update_mask: torch.Tensor, rm_update_weights: torch.Tensor,
                           use_rm_rl: bool) -> dict:

        metrics = {}
        if update_name == 'rm_update':
            batch, rewards, rm_update_weights = self._filter_batch_and_rewards_by_mask(batch, rm_update_mask, rewards, rm_update_weights)
        elif update_name == 'assistant_update' and (not use_rm_rl):
            num_active = rm_update_mask.sum()
            assert num_active > 0, "No active users"
            batch = batch.select_idxs(idxs=rm_update_mask)
            rewards = rewards[rm_update_mask]
        else:
            assert update_name == 'assistant_update', 'update_name must be rm_update or assistant_update'

        response_mask = batch.batch["response_mask"]
        token_level_scores = torch.zeros_like(response_mask, dtype=rewards.dtype)
        non_pad = response_mask.bool()
        rev_non_pad = torch.flip(non_pad, dims=[1])           # [B, T]
        # If there are multiple maximal values then the indices of the first maximal value are returned.
        rev_idx = rev_non_pad.float().argmax(dim=1)           # [B]  
        has_any = non_pad.any(dim=1)                          # [B] 
        last_idx = (response_mask.size(1) - 1) - rev_idx      # [B]
        batch_idx = torch.arange(response_mask.size(0), device=response_mask.device)
        token_level_scores[batch_idx[has_any], last_idx[has_any]] = rewards[has_any]
        batch.batch["token_level_scores"] = token_level_scores
        if update_name == 'rm_update':
            batch.batch["rm_update_weights"] = rm_update_weights.reshape(-1, 1)

        size_divisor = self.actor_rollout_wg.world_size
        batch, pad_size = pad_dataproto_to_divisor(batch, size_divisor)

        with marked_timer(f"{update_name}_prep", timing_raw):
            # Ensure response_mask exists
            assert "responses" in batch.batch
            
            # Compute global token count
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        
        with marked_timer(f"{update_name}_old_log_prob", timing_raw, color="blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            metrics["actor/entropy"] = entropy_agg.detach().item()
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            with marked_timer(f"{update_name}_ref", timing_raw, color="olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        if update_name == 'rm_update' and self.use_rm_critic:
            with marked_timer(f"{update_name}_values", timing_raw, color="cyan"):
                values = self.rm_critic_wg.compute_values(batch)
                batch = batch.union(values)
        
        if update_name == 'assistant_update' and self.use_policy_critic:
            with marked_timer(f"{update_name}_values", timing_raw, color="cyan"):
                values = self.policy_critic_wg.compute_values(batch)
                batch = batch.union(values)

        with marked_timer(f"{update_name}_adv", timing_raw, color="brown"):
            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            batch = compute_advantage(batch, 
                adv_estimator=self.config.algorithm.adv_estimator, 
                gamma=self.config.algorithm.gamma, 
                lam=self.config.algorithm.lam, 
                config=self.config.algorithm
                )

        # --- Model Updates ---
        if update_name == 'rm_update' and self.use_rm_critic:
            with marked_timer(f"{update_name}_update_critic", timing_raw, color="pink"):
                critic_output = self.rm_critic_wg.update_critic(batch)
            critic_metrics = reduce_metrics(critic_output.meta_info["metrics"])
            metrics.update(critic_metrics)

        if update_name == 'assistant_update' and self.use_policy_critic:
            with marked_timer(f"{update_name}_update_critic", timing_raw, color="pink"):
                critic_output = self.policy_critic_wg.update_critic(batch)
            critic_metrics = reduce_metrics(critic_output.meta_info["metrics"])
            metrics.update(critic_metrics)
        
        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer(f"{update_name}_update_actor", timing_raw, color="red"):
                # Make sure sft_update flag is not present for RL update
                batch.meta_info.pop("sft_update", None)
                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update(actor_metrics)

        # Prefix all collected metrics with the update_name
        prefixed_metrics = {f"{update_name}/{k}": v for k, v in metrics.items()}
        batch = unpad_dataproto(batch, pad_size)
        return prefixed_metrics, batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        for sub_task_idx in range(self.max_total_tasks):

            val_only = self.config.trainer.val_only
            if val_only and sub_task_idx < self.num_train_tasks: 
                assert self.config.personal.user_prefix_len == 0, "user_prefix_len must be 0 when val_only is True"
                progress_bar.update(len(self.train_dataloader)  * self.config.trainer.total_epochs)
                self.global_steps += len(self.train_dataloader) * self.config.trainer.total_epochs
                continue

            for epoch in range(self.config.trainer.total_epochs): # should be 1

                is_rag = self.config.personal.include_past_history
                if is_rag:
                    assert self.config.personal.user_prefix_len == 0, "RAG is not supported with user prefix"
                    assert self.config.personal.include_past_history_len >= 200, "RAG must have a large cache size"

                use_sft = self.config.personal.use_sft_loss
                use_rm_rl = self.config.personal.use_rm_loss
                use_assistant_rl = True
                if val_only or is_rag or sub_task_idx >= self.num_train_tasks:
                    use_sft, use_rm_rl, use_assistant_rl = False, False, False
                
                self.train_dataloader.load_state_dict(self._initial_dataloader_state)

                for user_batch_id, user_batch_data in enumerate(self.train_dataloader):

                    if self.global_steps >= self.total_training_steps: 
                        print("Global steps exceed total training steps, stop training")
                        break
                    extra_info_arr = user_batch_data["extra_info"]
                    sub_tasks_arr = np.array([ei["tasks"] for ei in extra_info_arr], dtype=object)

                    assert sub_tasks_arr is not None, "The field ['extra_info']['tasks'] is not present in Batch dict, so it's impossible to filter the user."
                    active_mask = np.array(
                        [sub_task_idx < len(sub_tasks) for sub_tasks in sub_tasks_arr],
                        dtype=bool,
                    )
                    if not active_mask.any():
                        continue

                    batch: DataProto = DataProto.from_single_dict(user_batch_data)
                    batch: DataProto = batch.select_idxs(idxs=active_mask)
                    sub_tasks_arr = sub_tasks_arr[active_mask]

                    gen_batch = self._get_gen_batch(batch)

                    metrics = {}
                    timing_raw = {}
                    active_user_logs = []

                    with marked_timer("step", timing_raw):
                        with marked_timer("gen", timing_raw, color="red"):
                            gen_batch = self._include_extra_data_for_gen_batch(gen_batch, sub_task_idx, self.all_user_states)

                            size_divisor = (
                                self.actor_rollout_wg.world_size
                                if not self.async_rollout_mode
                                else self.config.actor_rollout_ref.rollout.agent.num_workers
                            )
                            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, size_divisor)

                            if not self.async_rollout_mode:
                                gen_batch_output_padded, rm_gen_batch_output_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
                            else:
                                gen_batch_output_padded, rm_gen_batch_output_padded = self.async_rollout_manager.generate_sequences(gen_batch_padded)
                            timing_raw.update(gen_batch_output_padded.meta_info["timing"])
                            gen_batch_output_padded.meta_info.pop("timing", None)
                            rm_gen_batch_output_padded.meta_info.pop("timing", None)

                        gen_batch_output = unpad_dataproto(gen_batch_output_padded, pad_size)
                        rm_gen_batch_output = unpad_dataproto(rm_gen_batch_output_padded, pad_size)

                        rm_batch = copy.deepcopy(batch)
                        batch = batch.union(gen_batch_output)
                        rm_batch = rm_batch.union(rm_gen_batch_output)

                        if sub_task_idx == self.num_train_tasks:
                            print("[DONE]: record reset_len")
                            for i, user_id in enumerate(batch.non_tensor_batch['user_id']):
                                self.all_user_states[user_id].reset_len = len(self.all_user_states[user_id].history)
                        if sub_task_idx in self.config.personal.reset_step_ids:
                            print("[DONE]: truncate val history")
                            for i, user_id in enumerate(batch.non_tensor_batch['user_id']):
                                self.all_user_states[user_id].history = self.all_user_states[user_id].history[:self.all_user_states[user_id].reset_len]

                        if batch.non_tensor_batch['cannot_use'].any():
                            print(f"[NOTEIT] There exist {batch.non_tensor_batch['cannot_use'].sum()} item cannot be used, filter it!!!!!!!")
                        filter_mask = ~batch.non_tensor_batch['cannot_use']
                        if not filter_mask.any():
                            print("[NOTEIT] No valid data in this batch, skip!!!!!")
                            continue
                        batch = batch.select_idxs(idxs=filter_mask)
                        rm_batch = rm_batch.select_idxs(idxs=filter_mask)
                        sub_tasks_arr = sub_tasks_arr[filter_mask]

                        policy_rewards, real_rewards, estimated_rewards, completeness_rewards, real_personal_rewards, \
                            assistant_personal_rewards, feedback_mask, rm_update_rewards, reward_metrics, \
                            rm_update_weights = self._get_rewards_and_masks(batch, sub_task_idx)
                        metrics.update(reward_metrics)

                        for i, user_id in enumerate(batch.non_tensor_batch['user_id']):
                            curr_task_history = batch.non_tensor_batch['curr_task_history'][i]
                            self.all_user_states[user_id].history.extend(list(curr_task_history) if feedback_mask[i] else list(curr_task_history[:-1]))

                        if use_sft:
                            self.actor_rollout_wg.set_trainable_parameters(mode='p_vec_only')
                            do_zero = (user_batch_id==0)  # only zero grad for the first batch
                            batch.meta_info.update({"zero_grad": do_zero, "step_optimizer": False})
                            sft_metrics = self._perform_sft_update(batch, feedback_mask, timing_raw)
                            metrics.update(sft_metrics)

                        if use_rm_rl:
                            self.actor_rollout_wg.set_trainable_parameters(mode='all')
                            rm_batch.meta_info.update({"zero_grad": False, "step_optimizer": False})
                            rm_rl_metrics, rm_rl_batch = self._perform_rl_update(
                                rm_batch, "rm_update", timing_raw,
                                rm_update_rewards, feedback_mask, rm_update_weights, use_rm_rl
                            )
                            metrics.update(rm_rl_metrics)
                        else:
                            rm_rl_batch = rm_batch.select_idxs(idxs=feedback_mask)

                        if use_assistant_rl:
                            self.actor_rollout_wg.set_trainable_parameters(mode='model_only')
                            do_step = (user_batch_id == len(self.train_dataloader) - 1)
                            batch.meta_info.update({"zero_grad": False, "step_optimizer": do_step})
                            assistant_rl_metrics, assistant_rl_batch = self._perform_rl_update(
                                batch, "assistant_update", timing_raw,
                                policy_rewards, feedback_mask, None, use_rm_rl
                            )
                            metrics.update(assistant_rl_metrics)
                        else:
                            assistant_rl_batch = batch

                        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                        if rollout_data_dir:
                            with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                                inputs = self.tokenizer.batch_decode(assistant_rl_batch.batch["prompts"], skip_special_tokens=False)
                                outputs = self.tokenizer.batch_decode(assistant_rl_batch.batch["responses"], skip_special_tokens=False)
                                rm_input = self.tokenizer.batch_decode(rm_rl_batch.batch["prompts"], skip_special_tokens=False)
                                rm_output = self.tokenizer.batch_decode(rm_rl_batch.batch["responses"], skip_special_tokens=False)
                                history_till_now = [self.all_user_states[user_id].history for user_id in assistant_rl_batch.non_tensor_batch['user_id']]

                                mask_np = feedback_mask.cpu().numpy().astype(bool)
                                total_len = len(mask_np)
                                if use_rm_rl:
                                    rm_scores = rm_rl_batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                                    full_rm_scores = np.full(total_len, None, dtype=object)
                                    full_rm_scores[mask_np] = rm_scores
                                    full_rm_scores = full_rm_scores.tolist()
                                else:
                                    full_rm_scores = 0.0

                                if use_assistant_rl:
                                    assistant_scores = assistant_rl_batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                                else:
                                    assistant_scores = 0.0

                                full_rm_input = np.full(total_len, None, dtype=object)
                                full_rm_input[mask_np] = rm_input                
                                full_rm_input = full_rm_input.tolist()
                                full_rm_output = np.full(total_len, None, dtype=object)
                                full_rm_output[mask_np] = rm_output              
                                full_rm_output = full_rm_output.tolist()

                                curr_task_list = [task_list[sub_task_idx] for task_list in sub_tasks_arr]
                                origin_task_ids = [task['original_task_id'] for task in curr_task_list]

                                self._dump_generations(
                                    inputs=[re.sub(f'^({re.escape(self.tokenizer.pad_token)})+', '', x) for x in inputs],
                                    outputs=[re.sub(f'({re.escape(self.tokenizer.pad_token)})+$', '', x) for x in outputs],
                                    assistant_score=assistant_scores,
                                    rm_score=full_rm_scores,
                                    dump_path=rollout_data_dir,
                                    data_source=assistant_rl_batch.non_tensor_batch['data_source'],
                                    rm_input=[(re.sub(f'^({re.escape(self.tokenizer.pad_token)})+', '', x) if x is not None else None) for x in full_rm_input],
                                    rm_output=[(re.sub(f'({re.escape(self.tokenizer.pad_token)})+$', '', x) if x is not None else None) for x in full_rm_output],
                                    history_till_now=history_till_now,
                                    policy_reward=policy_rewards.cpu().tolist(),
                                    real_reward=real_rewards.cpu().tolist(),
                                    estimated_reward=estimated_rewards.cpu().tolist(),
                                    completeness_reward=completeness_rewards.cpu().tolist(),
                                    real_personal_reward=real_personal_rewards.cpu().tolist(),
                                    assistant_personal_reward=assistant_personal_rewards.cpu().tolist(),
                                    rm_update_reward=rm_update_rewards.cpu().tolist(),
                                    feedback_mask=mask_np.tolist(),
                                    sub_task_idx=sub_task_idx,
                                    user_id=assistant_rl_batch.non_tensor_batch['user_id'],
                                    origin_task_id=origin_task_ids,
                                    monitor_info=assistant_rl_batch.non_tensor_batch['monitor_info'],
                                )
                                for i, user_id in enumerate(assistant_rl_batch.non_tensor_batch['user_id']):
                                    curr_task_feedback = {
                                        "sub_task_idx": sub_task_idx,
                                        "origin_task_id": origin_task_ids[i],
                                        "policy_reward": policy_rewards.cpu().tolist()[i],
                                        "real_reward": real_rewards.cpu().tolist()[i],
                                        "estimated_reward": estimated_rewards.cpu().tolist()[i],
                                        "completeness_reward": completeness_rewards.cpu().tolist()[i],
                                        "real_personal_reward": real_personal_rewards.cpu().tolist()[i],
                                        "assistant_personal_reward": assistant_personal_rewards.cpu().tolist()[i],
                                        "rm_update_reward": rm_update_rewards.cpu().tolist()[i],
                                        "feedback_mask": mask_np.tolist()[i],
                                    }
                                    self.all_user_states[user_id].feedback_log.append(curr_task_feedback)
                    if self.global_steps % self.config.trainer.save_freq == 0:
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    # training metrics
                    metrics.update(
                        {
                            "training/global_step": self.global_steps,
                            "training/epoch": epoch,
                        }
                    )    
                    
                    if use_assistant_rl:
                        # Compute and log data-specific metrics for the Assistant update
                        assistant_data_metrics = compute_data_metrics(batch=assistant_rl_batch, use_critic=self.use_policy_critic)
                        metrics.update({f"assistant_update_{k}": v for k, v in assistant_data_metrics.items()})
                        metrics.update(compute_timing_metrics(batch=assistant_rl_batch, timing_raw=timing_raw, prefix='assistant_update'))

                    if use_rm_rl:
                        # Compute and log data-specific metrics for the RM update
                        rm_data_metrics = compute_data_metrics(batch=rm_rl_batch, use_critic=self.use_rm_critic)
                        metrics.update({f"rm_update_{k}": v for k, v in rm_data_metrics.items()})
                        metrics.update(compute_timing_metrics(batch=rm_rl_batch, timing_raw=timing_raw, prefix='rm_update'))

                    if use_assistant_rl and use_rm_rl:
                        n_gpus = self.resource_pool_manager.get_n_gpus()
                        metrics.update(compute_throughout_metrics(assistant_batch=assistant_rl_batch,
                                    rm_batch=rm_rl_batch, timing_raw=timing_raw, n_gpus=n_gpus))

                    logger.log(data=metrics, step=self.global_steps)

                    progress_bar.update(1)
                    self.global_steps += 1
            if (sub_task_idx % self.config.personal.save_all_user_states_freq == 0 and sub_task_idx != 0) or \
                                                (sub_task_idx == self.max_total_tasks - 1):
                self._save_all_user_states(sub_task_idx=sub_task_idx)  
        print("Training finished.")
        progress_bar.close()
        return