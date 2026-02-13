# coding=utf-8
#
# genearte_title2.py
#
#

import os
import logging

from typing import Optional
from my_llmkit.chat import UnifiedMessage
from .providers2 import get_provider_from_model_path


logger = logging.getLogger(__name__)


async def generate_title(input_messages: list[UnifiedMessage]) -> Optional[str]:
    model_path = os.getenv("GENERATE_TITLE_MODEL", "")
    if not model_path:
        logger.warning("GENERATE_TITLE_MODEL is not set")
        return
    (_, model_id) = model_path.split(":", 1)
    if not model_id:
        logger.error("GENERATE_TITLE_MODEL is invalid: %s", model_path)
        return

    # 根据 model_path 自动选择合适的 Provider
    provider = get_provider_from_model_path(model_path)
    if not provider:
        logger.error(f"Provider not found for model_path: {model_path}")
        return

    chat_client = provider.get_chat_client(model_id)

    instructions = "I will give you a conversation between a user and an LLM language model. You need to generate a concise and accurate title for this conversation that reflects the core content and theme. Please ensure the title is brief and to the point, avoiding lengthy descriptions. Output only the title, nothing else."

    prompts = ""
    for one_message in input_messages:
        if one_message.role not in ["user", "assistant"]:
            continue
        if not isinstance(one_message.content, str):
            continue
        prompts += f"{one_message.role}: {one_message.content}\n\n\n\n"

    if not prompts:
        logger.warning("No valid messages to generate title")
        return

    prompts = "以下是用户和语言模型的对话内容:\n\n\n\n" + prompts[:3000]

    messages: list[UnifiedMessage] = [
        UnifiedMessage(role="system", content=instructions),
        UnifiedMessage(role="user", content=prompts),
    ]

    result = await chat_client.run(messages=messages)
    output = result.last_content
    return output
