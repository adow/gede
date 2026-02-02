# coding=utf-8
"""
Model Information Management Module

This module handles fetching, caching, and providing access to LLM model metadata.
It aggregates information from multiple sources to provide comprehensive model capabilities,
pricing, and context window details.

IMPORTANT: All model identifiers use the full_model_id format: "provider:model"
    - provider: The model provider (e.g., "openai", "anthropic", "google")
    - model: The specific model name (e.g., "gpt-4", "claude-3-sonnet", "gemini-pro")
    - Examples: "openai:gpt-4", "anthropic:claude-3-5-sonnet", "google:gemini-1.5-pro"

Data Sources:
    - LiteLLM: Model pricing and context window information
    - Models.dev: Model capabilities and feature support
    - Custom URLs: Optional user-provided model information

Caching Strategy:
    - Model information is cached locally in /tmp/my_llmkit/cache/
    - Cache is automatically refreshed if older than 32 hours
    - First access triggers automatic cache initialization

Key Classes:
    ModelInfo: Pydantic model representing all metadata for a single LLM model
        - Includes: capabilities, pricing, token limits, provider info

Key Functions:
    get_model_info(full_model_id): Retrieve ModelInfo for a specific model
        - Parameter: full_model_id in "provider/model" format (e.g., "openai/gpt-4")
    read_model_info_dict(): Load all models from cache, refreshing if needed
        - Returns: Dict[full_model_id, ModelInfo]
    read_model_info_background(): Async task to pre-load cache in background
    update_model_info_cache(): Manually refresh cache from all sources

Usage:
    # Get information for a specific model (use full_model_id)
    model_info = await get_model_info("openai:gpt-4")
    print(f"Supports vision: {model_info.supports_vision}")
    print(f"Max tokens: {model_info.max_tokens}")

    # Pre-load cache in background
    read_model_info_background()

    # Get all cached models (keys are full_model_id)
    all_models = await read_model_info_dict()
    for full_model_id, info in all_models.items():
        print(f"{full_model_id}: {info.model_name}")
"""

#
import os
import logging
import json
import asyncio
from datetime import datetime
from typing import Optional, Literal, Any

import httpx
from pydantic import BaseModel, TypeAdapter

_cache_dir = "/tmp/my_llmkit/"


class ModelInfo(BaseModel):
    provider_id: str
    provider_name: Optional[str] = None

    model_id: str
    model_name: Optional[str] = None

    supports_tool_choice: Optional[bool] = None
    supports_function_calling: Optional[bool] = None
    supports_parallel_function_calling: Optional[bool] = None
    supports_vision: Optional[bool] = None
    supports_audio_input: Optional[bool] = None
    supports_pdf_input: Optional[bool] = None
    supports_audio_output: Optional[bool] = None
    supports_prompt_caching: Optional[bool] = None
    supports_response_schema: Optional[bool] = None
    supports_reasoning: Optional[bool] = None
    supports_web_search: Optional[bool] = None

    mode: Optional[
        str
        | Literal[
            "chat",
            "embedding",
            "completion",
            "image_generation",
            "video_generation",
            "audio_transcription",
            "audio_speech",
            "moderation",
            "rank",
        ]
    ] = None

    max_tokens: Optional[int] = None
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None

    input_cost_per_token: Optional[float] = None
    output_cost_per_token: Optional[float] = None
    output_const_per_reasoning_token: Optional[float] = None

    @property
    def supports_description(self) -> str:
        """返回模型支持的功能描述字符串"""
        features = []

        if self.supports_function_calling:
            features.append("Function Calling")
        if self.supports_tool_choice:
            features.append("Tool Choice")
        if self.supports_parallel_function_calling:
            features.append("Parallel Function Calling")
        if self.supports_vision:
            features.append("Vision")
        if self.supports_audio_input:
            features.append("Audio Input")
        if self.supports_pdf_input:
            features.append("PDF Input")
        if self.supports_audio_output:
            features.append("Audio Output")
        if self.supports_prompt_caching:
            features.append("Prompt Caching")
        if self.supports_response_schema:
            features.append("Structured Output")
        if self.supports_reasoning:
            features.append("Reasoning")
        if self.supports_web_search:
            features.append("Web Search")

        if not features:
            return ""

        return ", ".join(features)


# [model_full_id: model_info]
# model_full_id = provider_id/model_id
MODEL_INFO_DICT_CACHE: dict[str, ModelInfo] = {}
ModelInfoDictType = TypeAdapter(dict[str, ModelInfo])


def _cache_file():
    cache_dir = os.path.join(_cache_dir, "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    filename = os.path.join(cache_dir, "model_info_list.json")
    return filename


async def update_modep_info_from_litellm():
    global MODEL_INFO_DICT_CACHE
    url = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            logging.error(f"Failed to fetch model info list: {response.status_code}")
            return None
        rows = response.json()
        if not rows:
            logging.error("No data found in model info list.")
            return None

        del rows["sample_spec"]

        models_cache: dict[str, ModelInfo] = {}
        for model_full_id, model_dict in rows.items():
            parts = model_full_id.split("/")
            if len(parts) != 2:
                continue
            provider_id, model_id = parts
            # Convert to colon format for storage
            model_full_id = f"{provider_id}:{model_id}"
            provider_name = provider_id
            model_name = model_id
            supports_function_calling = model_dict.get(
                "supports_function_calling", None
            )
            supports_tool_choice = model_dict.get("supports_tool_choice", None)
            supports_parallel_function_calling = model_dict.get(
                "supports_parallel_function_calling", None
            )
            supports_vision = model_dict.get("supports_vision", None)
            supports_audio_input = model_dict.get("supports_audio_input", None)
            supports_pdf_input = model_dict.get("supports_pdf_input", None)
            supports_audio_output = model_dict.get("supports_audio_output", None)
            supports_prompt_caching = model_dict.get("supports_prompt_caching", None)
            supports_response_schema = model_dict.get("supports_response_schema", None)
            supports_reasoning = model_dict.get("supports_reasoning", None)
            supports_web_search = model_dict.get("supports_web_search", None)
            mode = model_dict.get("mode", None)
            max_tokens = model_dict.get("max_tokens", None)
            max_input_tokens = model_dict.get("max_input_tokens", None)
            max_output_tokens = model_dict.get("max_output_tokens", None)
            input_cost_per_token = model_dict.get("input_cost_per_token", None)
            output_cost_per_token = model_dict.get("output_cost_per_token", None)
            ouptput_const_per_reasoning_token = model_dict.get(
                "output_const_per_reasoning_token", None
            )
            models_cache[model_full_id] = ModelInfo(
                provider_id=provider_id,
                provider_name=provider_name,
                model_id=model_id,
                model_name=model_name,
                supports_function_calling=supports_function_calling,
                supports_tool_choice=supports_tool_choice,
                supports_parallel_function_calling=supports_parallel_function_calling,
                supports_vision=supports_vision,
                supports_audio_input=supports_audio_input,
                supports_pdf_input=supports_pdf_input,
                supports_audio_output=supports_audio_output,
                supports_prompt_caching=supports_prompt_caching,
                supports_response_schema=supports_response_schema,
                supports_reasoning=supports_reasoning,
                supports_web_search=supports_web_search,
                mode=mode,
                max_tokens=max_tokens,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                input_cost_per_token=input_cost_per_token,
                output_cost_per_token=output_cost_per_token,
                output_const_per_reasoning_token=ouptput_const_per_reasoning_token,
            )

        MODEL_INFO_DICT_CACHE.update(models_cache)


async def update_model_info_from_models_dev():
    global MODEL_INFO_DICT_CACHE
    url = "https://models.dev/api.json"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            logging.error(
                f"Failed to fetch models.dev info list: {response.status_code}"
            )
            return None
        result: dict[str, Any] = response.json()
        models_cache: dict[str, ModelInfo] = {}
        for provider_id, provider_dict in result.items():
            provider_name = provider_dict.get("name", "")
            models = provider_dict.get("models", [])
            for model_id, model_dict in models.items():
                model_name = model_dict.get("name", "")
                supports_function_calling = model_dict.get("tool_call", False)
                supports_reasoning = model_dict.get("reasoning", False)
                supports_response_schema = model_dict.get("structured_output", False)
                modalities = model_dict.get("modalities", {})
                modalities_input = modalities.get("input", [])
                supports_vision = "image" in modalities_input
                supports_pdf_input = "pdf" in modalities_input
                limit = model_dict.get("limit", {})
                max_tokens = limit.get("tokens", None)
                max_output_tokens = limit.get("output", None)
                model_full_id = f"{provider_id}:{model_id}"
                models_cache[model_full_id] = ModelInfo(
                    provider_id=provider_id,
                    provider_name=provider_name,
                    model_id=model_id,
                    model_name=model_name,
                    supports_function_calling=supports_function_calling,
                    supports_vision=supports_vision,
                    supports_pdf_input=supports_pdf_input,
                    supports_response_schema=supports_response_schema,
                    supports_reasoning=supports_reasoning,
                    mode="chat",
                    max_tokens=max_tokens,
                    max_output_tokens=max_output_tokens,
                )
        MODEL_INFO_DICT_CACHE.update(models_cache)


async def update_model_info_from_myllmkit(url: str):
    global MODEL_INFO_DICT_CACHE
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            logging.error(f"Failed to fetch model info list: {response.status_code}")
            return None
        rows = response.json()
        if not rows:
            logging.error("No data found in model info list.")
            return None

        models_cache: dict[str, ModelInfo] = ModelInfoDictType.validate_json(rows)
        if models_cache:
            MODEL_INFO_DICT_CACHE.update(models_cache)


async def update_model_info_cache(custom_url: Optional[str] = None):
    global MODEL_INFO_DICT_CACHE
    await update_modep_info_from_litellm()
    await update_model_info_from_models_dev()
    if custom_url:
        await update_model_info_from_myllmkit(custom_url)
    output = ModelInfoDictType.dump_python(MODEL_INFO_DICT_CACHE)
    with open(_cache_file(), "w") as f:
        f.write(json.dumps(output, indent=2, ensure_ascii=False))


async def read_model_info_dict():
    global MODEL_INFO_DICT_CACHE

    if MODEL_INFO_DICT_CACHE:
        return MODEL_INFO_DICT_CACHE

    filename = _cache_file()
    if not os.path.exists(filename):
        logging.debug("Model info cache file not found, creating new cache.")
        await update_model_info_cache()
    else:
        # Get the file's modification time
        file_mtime = os.path.getmtime(filename)
        current_time = datetime.now().timestamp()

        # Load existing cache first
        logging.debug("Loading model info from cache.")
        with open(filename, "r") as f:
            data = f.read()
            if not data:
                raise ValueError("Model info cache file is empty.")

            model_info_dict = ModelInfoDictType.validate_json(data)
            MODEL_INFO_DICT_CACHE = model_info_dict.copy()

        # Update cache in background if more than 32 hours have passed
        if current_time - file_mtime > 32 * 60 * 60:
            logging.debug(
                "Model info cache file is outdated, updating cache in background."
            )

            async def background_update():
                try:
                    await update_model_info_cache()
                    logging.debug("Background cache update completed successfully.")
                except Exception as e:
                    logging.exception(
                        f"Error updating model info cache in background: {e}"
                    )

            asyncio.create_task(background_update())

        return MODEL_INFO_DICT_CACHE

    logging.debug("Loading model info from cache.")

    # read from cache file
    with open(filename, "r") as f:
        data = f.read()
        if not data:
            raise ValueError("Model info cache file is empty.")

        model_info_dict = ModelInfoDictType.validate_json(data)
        MODEL_INFO_DICT_CACHE = model_info_dict.copy()
        return MODEL_INFO_DICT_CACHE


def read_model_info_background():
    async def wrapper():
        try:
            await read_model_info_dict()
        except Exception as e:
            logging.exception(f"Error reading model info dict in background: {e}")
            return None

    asyncio.create_task(wrapper())


async def get_model_info(full_model_id: str) -> Optional[ModelInfo]:
    models_cache = await read_model_info_dict()
    return models_cache.get(full_model_id)


# tests


async def test_read_model_info():
    models_cache = await read_model_info_dict()
    # print(
    #     json.dumps(
    #         ModelInfoDictType.dump_python(models_cache), indent=2, ensure_ascii=False
    #     )
    # )
    print(len(models_cache), "models loaded from cache.")


async def tests():
    await test_read_model_info()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(tests())
