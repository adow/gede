# coding=utf-8
#
# model_info.py
#
# Fetch model prices and capabilities from https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json,
# and update the cache file in a background thread when the program starts.
#
# However, the model names used here are not exactly the same as litellm, so some model information may not correspond.
#
import os

import json
import threading
from datetime import datetime
from typing import Optional, Literal, Any

import httpx
from pydantic import BaseModel, TypeAdapter


from ..top import logger, gede_dir

__all__ = [
    "ModelInfo",
    "read_model_info_dict",
    "MODEL_INFO_DICT_CACHE",
    "read_model_info_background",
]


class ModelInfo(BaseModel):
    supports_tool_choice: Optional[bool] = None
    supports_function_calling: Optional[bool] = None
    supports_parallel_function_calling: Optional[bool] = None
    supports_vision: Optional[bool] = None
    supports_audio_input: Optional[bool] = None
    supports_audio_output: Optional[bool] = None
    supports_prompt_caching: Optional[bool] = None
    supports_response_schema: Optional[bool] = None
    supports_reasoning: Optional[bool] = None
    supports_web_search: Optional[bool] = None

    litellm_provider: Optional[str] = None
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


MODEL_INFO_DICT_CACHE: dict[str, ModelInfo] = {}
ModelInfoDictType = TypeAdapter(dict[str, ModelInfo])


def _cache_file():
    cache_dir = os.path.join(gede_dir(), "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    filename = os.path.join(cache_dir, "model_info_list.json")
    return filename


def get_model_info_dict_from_cache():
    global MODEL_INFO_DICT_CACHE
    with open(_cache_file(), "r") as f:
        data = f.read()
        if not data:
            return None

        model_info_dict = ModelInfoDictType.validate_json(data)
        MODEL_INFO_DICT_CACHE = model_info_dict.copy()
        return model_info_dict


def update_model_info_dict_cache():
    url = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
    with httpx.Client() as client:
        response = client.get(url)
        if response.status_code != 200:
            logger.error(f"Failed to fetch model info list: {response.status_code}")
            return None
        rows = response.json()
        if not rows:
            logger.error("No data found in model info list.")
            return None

        del rows["sample_spec"]

        with open(_cache_file(), "w") as f:
            f.write(json.dumps(rows, indent=2, ensure_ascii=False))

        return get_model_info_dict_from_cache()


def read_model_info_dict():
    global MODEL_INFO_DICT_CACHE

    cache_dir = os.path.join(gede_dir(), "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    filename = os.path.join(cache_dir, "model_info_list.json")
    if not os.path.exists(filename):
        logger.debug("Model info cache file not found, creating new cache.")
        return update_model_info_dict_cache()

    # Get the file's modification time
    file_mtime = os.path.getmtime(filename)
    current_time = datetime.now().timestamp()

    # Update cache if more than 32 hours have passed
    if current_time - file_mtime > 32 * 60 * 60:
        logger.debug("Model info cache file is outdated, updating cache.")
        return update_model_info_dict_cache()

    logger.debug("Loading model info from cache.")
    return get_model_info_dict_from_cache()


def read_model_info_background():
    def wrapper():
        try:
            read_model_info_dict()
        except Exception as e:
            logger.exception(f"Error reading model info dict in background: {e}")
            return None

    threading.Thread(target=wrapper).start()


def get_model_info(model_id: str) -> Optional[ModelInfo]:
    return MODEL_INFO_DICT_CACHE.get(model_id)
