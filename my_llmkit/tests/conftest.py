import os
import logging
from pathlib import Path
from openai.types import Reasoning
from typing import Optional, Any
from datetime import datetime, timezone
from dotenv import load_dotenv

from my_llmkit.chat.model_settings import ModelSettings
from my_llmkit.chat import (
    OpenAICompatibleChatCompletion,
    ClaudeChatCompletion,
    ToolFunctions,
    UnifiedChunk,
    UnifiedMessage,
)

logger = logging.getLogger(__name__)


def load_env_file():
    env_filename = os.path.join(Path.home(), ".gede", "config", ".env")
    print(env_filename)
    load_dotenv(env_filename)


load_env_file()

# providers

api_key_openrouter = os.getenv("OPENROUTER_API_KEY", "")
api_base_openrouter = os.getenv("OPENROUTER_BASE_URL", "")

api_key_zenmux = os.getenv("ZENMUX_API_KEY", "")
api_base_zenmux_openai = os.getenv("ZENMUX_BASE_URL_OPENAI", "")
api_base_zenmux_anthropic = os.getenv("ZENMUX_BASE_URL_ANTHROPIC", "")

api_key_302ai = os.getenv("AI302_API_KEY", "")
api_base_302ai = os.getenv("AI302_BASE_URL", "")

api_key_deepseek = os.getenv("DEEPSEEK_API_KEY", "")
api_base_deepseek = os.getenv("DEEPSEEK_BASE_URL", "")

api_key_moonshot = os.getenv("MOONSHOT_API_KEY", "")
api_base_moonshot = os.getenv("MOONSHOT_BASE_URL", "")

api_key_doubao = os.getenv("ARK_API_KEY", "")
api_base_doubao = os.getenv("ARK_BASE_URL", "")

api_key_wenxin = os.getenv("QIANFAN_API_KEY", "")
api_base_wenxin = os.getenv("QIANFAN_BASE_URL", "")

api_key_qwen = os.getenv("DASHSCOPE_API_KEY", "")
api_base_qwen = os.getenv("DASHSCOPE_BASE_URL", "")

api_key_google = os.getenv("GOOGLE_API_KEY", "")
api_base_google = os.getenv("GOOGLE_BASE_URL", "")


# clients


def make_openai_client(
    api_key: str, api_base: str, model: str, reasoning: Optional[Reasoning] = None
):
    model_settings = ModelSettings(include_usage=True)
    if reasoning:
        model_setting_resolve = ModelSettings(reasoning=Reasoning(effort="medium"))
        model_settings = model_settings.resolve(model_setting_resolve)
    client = OpenAICompatibleChatCompletion(
        api_key=api_key,
        api_base=api_base,
        model=model,
        model_settings=model_settings,
    )
    return client


def make_claude_client(
    api_key: str, api_base: str, model: str, reasoning: Optional[bool] = None
):
    model_settings = ModelSettings(include_usage=True)
    if reasoning:
        model_settings.max_tokens = 30000
        model_settings.extra_body = {
            "thinking": {"type": "enabled", "budget_tokens": 10000}
        }
    client = ClaudeChatCompletion(
        api_key=api_key,
        api_base=api_base,
        model=model,
        model_settings=model_settings,
    )
    return client


def make_qwen_client(model: str, reasoning: Optional[bool] = None):
    model_settings = ModelSettings(include_usage=True)
    if reasoning:
        model_settings.extra_body = {
            "enable_thinking": True,
        }
    client = OpenAICompatibleChatCompletion(
        api_key=api_key_qwen,
        api_base=api_base_qwen,
        model=model,
        model_settings=model_settings,
    )
    return client


# models

gpt_5_2_zenmux = (
    api_key_zenmux,
    api_base_zenmux_openai,
    "openai/gpt-5.2",
)
gpt_5_2_openrouter = (
    api_key_openrouter,
    api_base_openrouter,
    "openai/gpt-5.2",
)

gemini_3_pro_openrouter = (
    api_key_openrouter,
    api_base_openrouter,
    "google/gemini-3-pro-preview",
)
gemini_3_pro_zenmux = (
    api_key_zenmux,
    api_base_zenmux_openai,
    "google/gemini-3-pro-preview",
)
gemini_3_flash_openrouter = (
    api_key_openrouter,
    api_base_openrouter,
    "google/gemini-3-flash-preview",
)
gemini_3_flash_zenmux = (
    api_key_zenmux,
    api_base_zenmux_openai,
    "google/gemini-3-flash-preview",
)
gemini_3_flash_google = (
    api_key_google,
    api_base_google,
    "gemini-3-flash-preview",
)


claude_4_5_sonnet_openrouter = (
    api_key_openrouter,
    api_base_openrouter,
    "anthropic/claude-sonnet-4.5",
)
claude_4_5_sonnet_zenmux = (
    api_key_zenmux,
    api_base_zenmux_anthropic,
    "anthropic/claude-sonnet-4.5",
)
claude_4_5_haiku_openrouter = (
    api_key_openrouter,
    api_base_openrouter,
    "anthropic/claude-haiku-4.5",
)
claude_4_5_haiku_zenmux = (
    api_key_zenmux,
    api_base_zenmux_anthropic,
    "anthropic/claude-haiku-4.5",
)
claude_4_5_haiku_302ai = (
    api_key_302ai,
    api_base_302ai,
    "claude-haiku-4-5-20251001",
)
kimi_k2_thinking_moonshot = (
    api_key_moonshot,
    api_base_moonshot,
    "kimi-k2-thinking",
)
kimi_k2_thinking_turbo_moonshot = (
    api_key_moonshot,
    api_base_moonshot,
    "kimi-k2-thinking-turbo",
)
kimi_k2_moonshot = (api_key_moonshot, api_base_moonshot, "kimi-k2-0905-preview")
kimi_k2_turbo_moonshot = (
    api_key_moonshot,
    api_base_moonshot,
    "kimi-k2-turbo-preview",
    None,
)
kimi_k2_5_moonshot = (api_key_moonshot, api_base_moonshot, "kimi-k2.5")
deepseek_chat_deepseek = (api_key_deepseek, api_base_deepseek, "deepseek-chat")
deepseek_reasoner_deepseek = (
    api_key_deepseek,
    api_base_deepseek,
    "deepseek-reasoner",
)
grok_4_1_fast_openrouter = (
    api_key_openrouter,
    api_base_openrouter,
    "x-ai/grok-4.1-fast",
)
grok_4_fast_openrouter = (
    api_key_openrouter,
    api_base_openrouter,
    "x-ai/grok-4-fast",
)
grok_4_openrouter = (api_key_openrouter, api_base_openrouter, "x-ai/grok-4")
grok_code_fast_1_openrouter = (
    api_key_openrouter,
    api_base_openrouter,
    "x-ai/grok-code-fast-1",
)
grok_4_1_fast_zenmux = (
    api_key_zenmux,
    api_base_zenmux_openai,
    "x-ai/grok-4.1-fast",
)
grok_4_fast_zenmux = (api_key_zenmux, api_base_zenmux_openai, "x-ai/grok-4-fast")
grok_4_zenmux = (api_key_zenmux, api_base_zenmux_openai, "x-ai/grok-4")
grok_code_fast_1_zenmux = (
    api_key_zenmux,
    api_base_zenmux_openai,
    "x-ai/grok-code-fast-1",
)
doubao_seed_1_8 = (api_key_doubao, api_base_doubao, "doubao-seed-1-8-251228")
doubao_seed_1_6 = (api_key_doubao, api_base_doubao, "doubao-seed-1-6-251015")
doubao_seed_2_pro = (api_key_doubao, api_base_doubao, "doubao-seed-2-0-pro-260215")
ernie_x_1_1 = (api_key_wenxin, api_base_wenxin, "ernie-x1.1-preview")
ernie_5_thinking = (api_key_wenxin, api_base_wenxin, "ernie-5.0-thinking-latest")
qwen_plus = ("qwen-plus", True)
