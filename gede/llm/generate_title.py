# coding= utf-8

import os
from typing import Optional

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    ModelSettings,
    Runner,
    TResponseInputItem,
)
from ..top import logger
from .providers import get_llm_model


MODEL = os.getenv("GENERATE_TITLE_MODEL", "")


async def generate_title(messages: list[TResponseInputItem]):
    if not MODEL:
        logger.warning("GENERATE_TITLE_MODEL is not set")
        return
    (provider_id, model_id) = MODEL.split(":", 1)
    if not provider_id or not model_id:
        logger.error("GENERATE_TITLE_MODEL is invalid: %s", MODEL)
        return
    model_result = get_llm_model(provider_id, model_id)

    agent = Agent(
        name="generate_title_agent",
        instructions="I will give you a conversation between a user and an LLM language model. You need to generate a concise and accurate title for this conversation that reflects the core content and theme. Please ensure the title is brief and to the point, avoiding lengthy descriptions. Output only the title, nothing else.",
        model=OpenAIChatCompletionsModel(
            model=model_result.model.model_id, openai_client=model_result.client
        ),
        model_settings=ModelSettings(
            include_usage=True,
            max_tokens=3000,
        ),
    )
    input_text = ""
    for one in messages[:5]:
        role = one.get("role", "user")
        content = one.get("content", "")
        input_text += f"{role}: {content}\n\n"
    logger.debug("generate_title input_text: %s", input_text)
    result = await Runner.run(agent, input_text)
    output: Optional[str] = result.final_output
    return output
