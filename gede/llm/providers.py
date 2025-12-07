# coding=utf-8
#
# manage all LLM providers
#

import os
import asyncio
import concurrent.futures
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional

from agents import ModelSettings
from pydantic import BaseModel, Field, TypeAdapter


from openai import AsyncClient

from ..top import logger, gede_dir
from .llm_provider import LLMProviderBase, LLMModel
from .voice_engine_provider import VoiceEngineProvider
from .openrouter_provider import OpenRouterProvider
from .ai302_provider import AI302Provider
from .kimi_provider import KimiProvider
from .deepseek_provider import DeepSeekProvider
from .wenxin_provider import WenxinProvider
from .qwen_provider import QwenProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider

DEFAULT_MODEL_PATH = "voice_engine:doubao-seed-1-6"


def gede_cache_dir():
    return os.path.join(gede_dir(), "cache")


PROVIDERS: list[LLMProviderBase] = [
    VoiceEngineProvider(),
    OpenRouterProvider(),
    AI302Provider(),
    KimiProvider(),
    DeepSeekProvider(),
    WenxinProvider(),
    QwenProvider(),
    OpenAIProvider(),
    AnthropicProvider(),
]


async def load_all_providers_models_async():
    """Load all providers models concurrently"""
    tasks = []
    for provider in PROVIDERS:
        task = provider.load_models_async()
        tasks.append(task)

    # Execute all tasks concurrently
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.debug("All providers models loaded.")


def load_all_providers_models_sync():
    """Load all providers models concurrently using thread pool (synchronous version)"""
    logger.info("Loading all providers models...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all tasks to thread pool
        futures = []
        for provider in PROVIDERS:
            future = executor.submit(provider.load_models)  # Call synchronous method
            futures.append(future)

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)

    logger.debug("All providers models loaded.")


# cache
#
class ModelCache(BaseModel):
    model_id: str
    name: str


class ProviderCache(BaseModel):
    provider_id: str
    name: str
    models: list[ModelCache] = []


def update_all_providers_models_cache():
    folder = gede_cache_dir()
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(gede_cache_dir(), "llm_providers_models.json")
    load_all_providers_models_sync()
    cache_data: list[ProviderCache] = []
    for one_provider in PROVIDERS:
        provider_cache = ProviderCache(
            provider_id=one_provider.provider_id,
            name=one_provider.name,
        )
        models_cache: list[ModelCache] = []
        for one_model in one_provider.models:
            models_cache.append(
                ModelCache(
                    model_id=one_model.model_id,
                    name=one_model.name,
                )
            )
        provider_cache.models = models_cache
        cache_data.append(provider_cache)
    with open(filename, "w", encoding="utf-8") as f:
        ProviderCacheListType = TypeAdapter(list[ProviderCache])
        f.write(ProviderCacheListType.dump_json(cache_data).decode("utf-8"))
        logger.debug("LLM providers models cache updated.")


def read_all_providers_models_sync(no_cache=False):
    if no_cache:
        return update_all_providers_models_cache()
    folder = gede_cache_dir()
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(gede_cache_dir(), "llm_providers_models.json")
    # Check if file exists and needs updating
    if not os.path.exists(filename):
        # File does not exist, create cache
        return update_all_providers_models_cache()

    # Get the file's modification time
    file_mtime = os.path.getmtime(filename)
    current_time = datetime.now().timestamp()

    # Update cache if more than 24 hours have passed
    if current_time - file_mtime > 24 * 60 * 60:
        return update_all_providers_models_cache()

    ProviderCacheListType = TypeAdapter(list[ProviderCache])
    with open(filename, "r", encoding="utf-8") as f:
        cache_data = ProviderCacheListType.validate_json(f.read())
        if not cache_data:
            return update_all_providers_models_cache()
        logger.debug("LLM providers models cache loading.")
        for one_provider in cache_data:
            find_providers = [
                one for one in PROVIDERS if one.provider_id == one_provider.provider_id
            ]
            if find_providers:
                provider = find_providers[0]
                provider.models = [
                    LLMModel(model_id=one_model.model_id, name=one_model.name)
                    for one_model in one_provider.models
                ]
        logger.debug("LLM providers models cache loaded.")


@dataclass
class GetLLMModelResult:
    provider_id: str
    provider_name: str
    client: AsyncClient

    model: LLMModel
    provider: LLMProviderBase

    # List of features supported by the model
    model_supports: Optional[list[str]] = field(default_factory=list)


def get_llm_model(provider_id: str, model_id: str) -> GetLLMModelResult:
    """Get the LLM model for the specified provider and model ID"""
    for provider in PROVIDERS:
        if provider.provider_id == provider_id:
            for model in provider.models:
                if model.model_id == model_id:
                    supports: list[str] = []
                    supports_reasoning = provider.supports_reasoning_for_model_id(
                        model_id
                    )
                    if supports_reasoning:
                        supports.append("Reasoning")
                    supports_web_search = provider.supports_web_search_for_model_id(
                        model_id
                    )
                    if supports_web_search:
                        supports.append("Web Search")

                    if provider.supports_tools_for_model_id(model_id):
                        supports.append("Tools")
                    return GetLLMModelResult(
                        provider_id=provider.provider_id,
                        provider_name=provider.name,
                        client=provider.client,
                        model=model,
                        model_supports=supports,
                        provider=provider,
                    )
    raise Exception(f"no such model found: {provider_id}, {model_id}")


def get_llm_model_settings(provider_id: str, model_id: str) -> ModelSettings:
    for provider in PROVIDERS:
        if provider.provider_id == provider_id:
            for model in provider.models:
                if model.model_id == model_id:
                    return provider.model_settings_for_model_id(model_id)

    return ModelSettings()


PATH_LIST: list[str] = []
PATH_VALUE_LIST: list[tuple[str, str]] = []


def get_model_path_value_list(no_cache=False):
    """
    Get the table of model paths and values
    Returns:
        - name: provider name + model name
        - value (which is model_path): provider_id + model_id
    """
    global PATH_VALUE_LIST
    if PATH_VALUE_LIST and not no_cache:
        return PATH_VALUE_LIST
    read_all_providers_models_sync(no_cache)
    for one_provider in PROVIDERS:
        for one_model in one_provider.models:
            name = f"{one_provider.name}:{one_model.name}"
            supports: list[str] = []
            if one_provider.supports_reasoning_for_model_id(one_model.model_id):
                supports.append("Reasoning")
            if one_provider.supports_web_search_for_model_id(one_model.model_id):
                supports.append("Web Search")
            if one_provider.supports_tools_for_model_id(one_model.model_id):
                supports.append("Tools")
            if supports:
                name += " (Supports: " + ", ".join(supports) + ")"
            PATH_VALUE_LIST.append(
                (
                    name,
                    f"{one_provider.provider_id}:{one_model.model_id}",
                )
            )
    return PATH_VALUE_LIST
