# coding=utf-8
"""
Models Package - Model Information and Capability Detection

This package provides utilities for querying LLM model information and detecting their capabilities.
It automatically fetches and caches model metadata from multiple sources (litellm, models.dev, etc.)
and provides convenient async functions to check what features each model supports.

IMPORTANT: All model_id parameters require the full_model_id format: "provider/model"
    Examples: "openai/gpt-4", "anthropic/claude-3-sonnet", "google/gemini-pro"

Usage:
    # Check if a model supports specific capabilities
    from my_llmkit.models import (
        supports_reasoning,
        supports_vision,
        supports_function_calling,
        get_model_info
    )
    
    # Check individual capabilities (use full_model_id format)
    if await supports_reasoning("openai/gpt-4"):
        print("Model supports reasoning")
    
    if await supports_vision("anthropic/claude-3-sonnet"):
        print("Model supports vision")
    
    # Get full model information (use full_model_id format)
    model_info = await get_model_info("openai/gpt-4")
    if model_info:
        print(f"Max tokens: {model_info.max_tokens}")
        print(f"Cost per token: {model_info.input_cost_per_token}")

Key Features:
    - Automatic model metadata caching with auto-refresh
    - Support detection for: reasoning, vision, function calling, response schema, web search, prompt caching
    - Unified interface for multiple model providers
    - Async-first API design
"""
#

from .info import (
    ModelInfo,
    ModelInfoDictType,
    read_model_info_dict,
    read_model_info_background,
    get_model_info,
)
from .capabilities import (
    supports_reasoning,
    supports_vision,
    supports_function_calling,
    supports_response_schema,
    supports_web_search,
    supports_prompt_caching,
)

__all__ = [
    "ModelInfo",
    "ModelInfoDictType",
    "read_model_info_dict",
    "read_model_info_background",
    "get_model_info",
    "supports_reasoning",
    "supports_vision",
    "supports_function_calling",
    "supports_response_schema",
    "supports_web_search",
    "supports_prompt_caching",
]
