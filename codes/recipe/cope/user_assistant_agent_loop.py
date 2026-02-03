# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2024 ModelBest Inc. and/or its affiliates
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

import json
import re
import random
from typing import Any, List, Optional
from recipe.cope.api_request_async import async_request
from recipe.cope.agent_loop import AgentLoopBase, AgentLoopOutput
from recipe.cope.chat_model import ChatModel, MaxTokenExceededError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage, convert_to_messages, convert_to_openai_messages
from recipe.cope.rag_utils import get_rag_result
from recipe.cope.prompt_utils import (
    get_completeness_eval_prompt,
    get_assistant_self_eval_prompt,
    parse_assistant_self_eval,
    get_conversation_history,
    get_user_eval_prompt,
    parse_user_eval_response,
    parse_completeness_response,
    render_system_prompt,
    render_system_prompt_rag
    )

class PersonalAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True

        print("Performing class-level PersonalAgentLoop initialization")

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> List[AgentLoopOutput]:
        
        # 1. --- Initialization ---
        user_id: str = kwargs['extra_info']["user_id"]
        initial_messages: list[dict] = list(kwargs['extra_info']["prompt"])
        profile: dict = kwargs['extra_info']['profile']
        tasks: list[dict[str, Any]] = kwargs['extra_info']['tasks']
        sub_task_id: int = kwargs["extra_info"]["sub_task_idx"]
        task: dict[str, Any] = tasks[sub_task_id] 
        apikey: str = kwargs["ude_api_key"]
        user_prefix_len: int = kwargs['user_prefix_len']
        async_request_mode: str = self.config.personal.user.async_request_mode
        rollout_config = self.config.actor_rollout_ref.rollout

        data_idx = kwargs['extra_info']["data_idx"]
        use_real_feedback = random.random() < self.config.personal.real_reward_usage_prob

        model = ChatModel(
            model=self.config.actor_rollout_ref.model_type,
            client=self.server_manager,
            tokenizer=self.tokenizer,
            max_tokens=self.config.personal.rollout.assistant_max_response_len,
        )

        user_call_failed = False
        user_eval_call_failed = False
        completeness_eval_call_failed = False
        exceed_flag = 0
        user_parse_failed = False
        completeness_parse_failed = False
        rm_parse_failed = False
        # 2. --- Conversation Simulation ---
        messages: list[BaseMessage] = convert_to_messages(initial_messages)
        prompt_message_length = len(messages)
        if prompt_message_length == 1:
            system_message: SystemMessage = render_system_prompt(user_id)
            messages.append(HumanMessage(content=task['initial query']))
        else:
            # Need to implement by self
            past_history: str = get_conversation_history(messages[1:])
            rag_result = get_rag_result(past_history, task['initial query'])
            system_message: SystemMessage = render_system_prompt_rag(rag_result)
            messages.append(HumanMessage(content=task['initial query']))

        assistant_response = await model.agenerate([[system_message, messages[-1]]], sampling_params=sampling_params, enable_thinking=False, 
                                turn_id=0, sub_task_id=sub_task_id, user_id=user_id, user_prefix_len=user_prefix_len) 
        
        assistant_response = assistant_response.generations[0][0]
        messages.append(assistant_response.message)

        curr_task_history = messages[prompt_message_length:]
        assert len(curr_task_history) == 2, "curr_task_history must have two messages"
        curr_task_history_str = get_conversation_history(curr_task_history)
        user_eval_prompt = get_user_eval_prompt(task["rubrics"], curr_task_history_str)
        user_response = await async_request(user_eval_prompt, apikey, async_request_mode)
        if user_response:
            eval_str, user_personalization_score, parse_failed = parse_user_eval_response(user_response)
            user_parse_failed = parse_failed
            messages.append(HumanMessage(content=eval_str))
        else:
            print("[Warning] calling user eval failed, return None!")
            eval_str, user_personalization_score = "None", 0.0
            user_eval_call_failed, user_parse_failed = True, True
            messages.append(HumanMessage(content=eval_str))
        
        completeness_eval_prompt = get_completeness_eval_prompt(curr_task_history_str)
        completeness_response = await async_request(completeness_eval_prompt, apikey, async_request_mode)
        if completeness_response:
            reasoning, completeness_score, parse_failed = parse_completeness_response(completeness_response)
            completeness_parse_failed = parse_failed
        else:
            print("[Warning] calling completeness eval failed, return None!")
            reasoning, completeness_score = "None", 0.0
            completeness_eval_call_failed, completeness_parse_failed = True, True
        
        assert messages[0].type == "system", "First message must be system"
        assert messages[-1].type == "human", "Last message must be human"
        assert len(messages) - prompt_message_length > 1, "New task only have the initial user query"

        for i in range(len(messages) - 1, -1, -1):
            if messages[i].type == "ai":
                break  

        last_ai_message = messages[i]
        assert "prompt_ids" in last_ai_message.response_metadata, "Last ai message must have prompt_ids in response_metadata"
        assert "response_mask" in last_ai_message.response_metadata, (
            "Last ai message must have response_mask in response_metadata"
        )
        prompt_ids = last_ai_message.response_metadata.pop("prompt_ids")
        response_mask = last_ai_message.response_metadata.pop("response_mask")
        
        human_responses = messages[i + 1 :]    
        human_response_ids =  self.tokenizer.encode('\n') + self.tokenizer.apply_chat_template(
                convert_to_openai_messages(human_responses), add_generation_prompt=False, tokenize=True
            )
        prompt_ids += human_response_ids
        response_mask += [0] * len(human_response_ids)

        response_ids = prompt_ids[-len(response_mask) :]
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]     

        # 3. --- Evaluation Phase ---
        # Call Assistant for rating (with thinking enabled)
        prompt_a_system, prompt_a_user = get_assistant_self_eval_prompt(
            curr_task_history_str, user_id, 
            user_eval_reasoning=eval_str if use_real_feedback else None
        )

        prompt_b_system, prompt_b_user = get_assistant_self_eval_prompt(
            curr_task_history_str, user_id, 
            user_eval_reasoning=None
        )
        assist_reward_generate_messages = [[SystemMessage(content=prompt_b_system), HumanMessage(content=prompt_b_user)]]
        use_note_in_reward = False
        if use_real_feedback and random.random() < self.config.personal.real_reward_user_note_prob:
            assist_reward_generate_messages = [[SystemMessage(content=prompt_a_system), HumanMessage(content=prompt_a_user)]]
            use_note_in_reward = True
        personalization_resp_assistant = await model.agenerate(
            assist_reward_generate_messages,
            sampling_params=sampling_params,
            enable_thinking=True,
            turn_id=None,
            sub_task_id=sub_task_id,
            user_id=user_id,
            user_prefix_len=user_prefix_len
        )
        personalization_resp_assistant = personalization_resp_assistant.generations[0][0].message
        rm_prompt_ids = personalization_resp_assistant.response_metadata.pop("prompt_ids")
        rm_response_mask = personalization_resp_assistant.response_metadata.pop("response_mask")
        assert rm_response_mask == [1] * len(rm_response_mask), "rm_response_mask must be all 1"

        if use_note_in_reward:
            rm_prompt_b_ids = self.tokenizer.apply_chat_template(
                convert_to_openai_messages([SystemMessage(content=prompt_b_system), HumanMessage(content=prompt_b_user)]),
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=True
            )
            session_id: int = sub_task_id
            user_id_int: int = int(user_id.replace('user', ''))
            rm_prompt_b_ids = [session_id] + [user_id_int]*(user_prefix_len-1) + rm_prompt_b_ids

            rm_response_ids = rm_prompt_ids[-len(rm_response_mask) :]
            rm_prompt_ids = rm_prompt_b_ids 
        else:
            rm_response_ids = rm_prompt_ids[-len(rm_response_mask) :]
            rm_prompt_ids = rm_prompt_ids[: len(rm_prompt_ids) - len(rm_response_mask)]     

        personalization_resp_assistant_str = personalization_resp_assistant.content
        assert self.tokenizer.eos_token not in personalization_resp_assistant_str, "personalization_resp_assistant_str must not contain <|im_end|>"
        assistant_personalization_score, parse_failed = parse_assistant_self_eval(personalization_resp_assistant_str)
        rm_parse_failed = parse_failed
        cannot_use = user_call_failed or user_eval_call_failed or user_parse_failed or \
            completeness_eval_call_failed or completeness_parse_failed or rm_parse_failed

        # 4. --- Prepare Output ---
        return [AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            curr_task_history=convert_to_openai_messages(messages[prompt_message_length:]),
            num_turns=len([msg for msg in messages[prompt_message_length:] if msg.type == "ai"]),
            metrics={},
            user_completeness_score=completeness_score,
            user_personalization_score=user_personalization_score,
            assistant_personalization_score=assistant_personalization_score,
            user_id=user_id,
            cannot_use=cannot_use,
            feedback_mask=use_real_feedback,
            monitor_info={
                "exceed_flag": exceed_flag, 
                "user_call_failed": user_call_failed,
                "user_eval_call_failed": user_eval_call_failed,
                "completeness_eval_call_failed": completeness_eval_call_failed,
                "completeness_parse_failed": completeness_parse_failed,
                "user_parse_failed": user_parse_failed,
                "rm_parse_failed": rm_parse_failed,
                # "initial_query_prompt": initial_query_prompt,
                "initial_query_response": task['initial query'],
                "user_eval_prompt": user_eval_prompt,
                "user_eval_response": user_response,
                "completeness_eval_prompt": completeness_eval_prompt,
                "completeness_eval_response": completeness_response,
                "assistant_self_eval_system_prompt_a": prompt_a_system,
                "assistant_self_eval_user_prompt_a": prompt_a_user,
                "assistant_self_eval_system_prompt_b": prompt_b_system,
                "assistant_self_eval_user_prompt_b": prompt_b_user,
                "assistant_self_eval_response": personalization_resp_assistant_str,
                "use_note_in_reward": use_note_in_reward,
            }

        ), AgentLoopOutput(
            prompt_ids=rm_prompt_ids,
            response_ids=rm_response_ids,
            response_mask=rm_response_mask,
            curr_task_history=None,
            num_turns=len([msg for msg in messages[prompt_message_length:] if msg.type == "ai"]),
            metrics={},
            user_completeness_score=completeness_score,
            user_personalization_score=user_personalization_score,
            assistant_personalization_score=assistant_personalization_score,
            user_id=user_id,
            cannot_use=None,
            feedback_mask=None,
            monitor_info=None
        )]

