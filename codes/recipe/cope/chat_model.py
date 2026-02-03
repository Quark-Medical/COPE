# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Ref: https://python.langchain.com/docs/how_to/custom_chat_model/
"""

import asyncio
import logging
import os
import uuid
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    convert_to_openai_messages,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import Field

from recipe.cope.agent_loop import AgentLoopOutput, AsyncLLMServerManager
from copy import deepcopy


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

async def decode_assistant_response(responses_ids: list[int], tokenizer) -> str:
    loop = asyncio.get_running_loop()
    content = await loop.run_in_executor(None, tokenizer.decode, responses_ids)

    return content

class MaxTokenExceededError(Exception):
    """Indicate that history chat messages + human message exceeds LLM max_tokens."""

    pass


class ChatModel(BaseChatModel):
    model_name: str = Field(alias="model")
    """The name of the model"""

    client: AsyncLLMServerManager
    """AsyncLLM server manager"""

    tokenizer: Any
    """Tokenizer for the model"""

    max_tokens: int
    """Max tokens to generate"""

    temperature: float = 1.0
    """Temperature for sampling"""

    top_p: float = 1.0
    """Top p for sampling"""

    repetition_penalty: float = 1.0
    """Repetition penalty for sampling"""


    def with_structured_output(
        self,
        schema: dict | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseChatModel]:
        """Ref: https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/"""
        raise NotImplementedError

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:

        request_id, prompt_ids, response_mask = await self.preprocess(messages, **kwargs)

        sampling_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }
        if "sampling_params" in kwargs:
            sampling_params.update(kwargs["sampling_params"])

        if kwargs['turn_id'] == None:
            output = await self.client.generate(
                request_id=request_id, prompt_ids=prompt_ids, sampling_params={**sampling_params, "max_new_tokens": 2048}
            )
        else:
            output = await self.client.generate(
                request_id=request_id, prompt_ids=prompt_ids, sampling_params={**sampling_params, "max_new_tokens": 768}
            )

        message = await self._postprocess(request_id, prompt_ids, response_mask, output.token_ids, **kwargs)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return self.model_name

    async def preprocess(self, messages: list[BaseMessage], **kwargs: Any) -> tuple[str, list[int], list[int]]:

        # messages: system prompt, past_task_history, human, ai, human, ai, human
        if kwargs['turn_id'] != None:
            assert messages[-1].type == "human", (f"Last message must be human, but got {messages[-1].type}")
        loop = asyncio.get_running_loop()

        # Case 1: initial chat completion: [system], human
        if (messages[-1].type == "human" and kwargs['turn_id'] == 0) or kwargs['turn_id'] == None:
            prompt_ids = await loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        convert_to_openai_messages(messages),
                        add_generation_prompt=True,
                        tokenize=True,
                        enable_thinking=kwargs['enable_thinking']
                    ),
                )
            session_id: int = int(kwargs['sub_task_id'])
            user_id: int = int(kwargs['user_id'].replace('user', ''))
            user_prefix_len: int = int(kwargs['user_prefix_len'])
            if user_prefix_len > 0:
                prompt_ids = [session_id] + [user_id]*(user_prefix_len-1) + prompt_ids
            return str(uuid.uuid4()), prompt_ids, []

        # Case 2: follow up chat completion with human response: system prompt, past_task_history, human, ai, human, ai, human
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].type == "ai":
                break
        if "prompt_ids" not in messages[i].response_metadata:
            print('messages[i].response_metadata', messages[i].response_metadata)
            print('messages types', [x.type for x in messages], i)
        assert "prompt_ids" in messages[i].response_metadata, "Last message must have prompt_ids in response_metadata"
        assert "response_mask" in messages[i].response_metadata, ("Last message must have response_mask in response_metadata")

        # encode human response
        human_responses = convert_to_openai_messages(messages[i + 1 :])
        human_response_ids = await loop.run_in_executor(
            None,
            lambda messages=human_responses: self.tokenizer.encode('\n') + self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, enable_thinking=kwargs['enable_thinking']
            ),
        )

        # stop generation if response length exceeds max response length
        if len(messages[i].response_metadata["response_mask"]) + len(human_response_ids) >= self.max_tokens:
            raise MaxTokenExceededError(f"Max response length {self.max_tokens} exceeded")

        # append human response to prompt
        request_id = messages[i].response_metadata.pop("request_id")
        prompt_ids = messages[i].response_metadata.pop("prompt_ids")
        response_mask = messages[i].response_metadata.pop("response_mask")
        prompt_ids += human_response_ids
        response_mask += [0] * len(human_response_ids)

        return request_id, prompt_ids, response_mask

    async def _postprocess(
        self, request_id: str, prompt_ids: list[int], response_mask: list[int], response_ids: list[int], **kwargs: Any
    ) -> AIMessage:
        eos_id = self.tokenizer.eos_token_id
        if response_ids[-1] != eos_id:
            response_ids.append(eos_id)

        new_prompt_ids = prompt_ids + response_ids
        new_response_mask = response_mask + [1] * len(response_ids)
        content = await decode_assistant_response(response_ids, self.tokenizer)

        message = AIMessage(
            content=content.replace(self.tokenizer.eos_token, ""),
            response_metadata={
                "request_id": request_id,
                "prompt_ids": new_prompt_ids,
                "response_mask": new_response_mask,
            },
        )
        return message
