# coding=utf-8
"""
Model Capabilities Detection Module

This module provides convenient async functions to check whether specific LLM models
support various capabilities. All functions return boolean values indicating feature support.

IMPORTANT: All model_id parameters require the full_model_id format: "provider/model"
    Examples: "openai/gpt-4", "anthropic/claude-3-sonnet", "google/gemini-pro"

Available Capability Checks:
    - supports_reasoning: Check if model supports extended reasoning/thinking
    - supports_vision: Check if model can process image inputs
    - supports_function_calling: Check if model supports tool/function calling
    - supports_response_schema: Check if model supports structured output/JSON schema
    - supports_web_search: Check if model has built-in web search capabilities
    - supports_prompt_caching: Check if model supports prompt caching for cost optimization

Usage:
    from my_llmkit.models import supports_vision, supports_function_calling
    
    # Check if a model supports vision (use full_model_id format)
    if await supports_vision("anthropic/claude-3-sonnet"):
        print("This model can process images")
    
    # Check multiple capabilities
    model_id = "openai/gpt-4"
    has_vision = await supports_vision(model_id)
    has_tools = await supports_function_calling(model_id)
    
    if has_vision and has_tools:
        print("Model supports both vision and function calling")

Return Values:
    All functions return False if:
        - Model not found in cache
        - Model doesn't support the capability
        - Capability information is not available
"""

from .info import read_model_info_dict, get_model_info


async def supports_reasoning(model_id: str) -> bool:
    """
    检测模型是否支持推理能力
    """
    model = await get_model_info(model_id)
    if not model:
        return False
    return model.supports_reasoning if model.supports_reasoning is not None else False


async def supports_vision(model_id: str) -> bool:
    """
    检测模型是否支持视觉能力
    """
    model = await get_model_info(model_id)
    if not model:
        return False
    return model.supports_vision if model.supports_vision is not None else False


async def supports_function_calling(model_id: str) -> bool:
    """
    检测模型是否支持函数调用
    """
    model = await get_model_info(model_id)
    if not model:
        return False
    return (
        model.supports_function_calling
        if model.supports_function_calling is not None
        else False
    )


async def supports_response_schema(model_id: str) -> bool:
    """
    检测模型是否支持结构化输出
    """
    model = await get_model_info(model_id)
    if not model:
        return False
    return (
        model.supports_response_schema
        if model.supports_response_schema is not None
        else False
    )


async def supports_web_search(model_id: str) -> bool:
    """
    检测模型是否支持网页搜索
    """
    model = await get_model_info(model_id)
    if not model:
        return False
    return model.supports_web_search if model.supports_web_search is not None else False


async def supports_prompt_caching(model_id: str) -> bool:
    """
    检测模型是否支持提示缓存
    """
    model = await get_model_info(model_id)
    if not model:
        return False
    return (
        model.supports_prompt_caching
        if model.supports_prompt_caching is not None
        else False
    )
