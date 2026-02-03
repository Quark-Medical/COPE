import random
import torch
from typing import Dict, List
from collections import defaultdict
from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager


class PersonalRewardManager(AbstractRewardManager):
    def __init__(
        self, 
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "None",
        reward_coefficients: Dict = None,
        **kwargs,
    ) -> None:
        pass

    def __call__(self, data: DataProto, return_dict: bool = False):
        pass