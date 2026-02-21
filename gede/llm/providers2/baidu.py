# coding=utf-8
#
#

import os
import logging
from typing import Optional
import httpx

from my_llmkit.models import get_model_info
from my_llmkit.chat.model_settings import ModelSettings
from my_llmkit.chat import LLMChatCompletion, OpenAICompatibleChatCompletion

from .base import LLMProviderBase
from .reasoning import ReasoningEffortType

logger = logging.getLogger(__name__)

API_KEY = os.getenv("WENXIN_API_KEY", "")
API_BASE_URL = os.getenv("WENXIN_BASE_URL", "https://qianfan.baidubce.com/v2")


class BaiduProvider(LLMProviderBase):
    def __init__(self):
        super().__init__(
            provider_id="baidu",
            name="Baidu",
        )

    def get_chat_client(
        self, model_id: str, model_settings: Optional[ModelSettings] = None
    ) -> LLMChatCompletion:
        return OpenAICompatibleChatCompletion(
            api_key=API_KEY,
            model=model_id,
            api_base=API_BASE_URL,
            model_settings=model_settings,
        )

    @property
    def default_models(self) -> Optional[list[str]]:
        # 百度云千帆当前旗舰模型（推理 + 通用）
        return ["ernie-5.0-thinking-preview", "ernie-4.5-turbo-128k"]

    async def load_models(self):
        if not API_KEY:
            logger.warning("Baidu API key is not set, skipping model loading.")
            return
        url = f"{API_BASE_URL}/models"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers, timeout=10)
                if response.status_code != 200:
                    logger.error(
                        f"Baidu load models failed: {response.status_code}, {response.text}"
                    )
                    return
                result = response.json()
                data = result.get("data", [])
                for one in data:
                    model_id = one.get("id", "").lower()
                    if not model_id or not model_id.startswith("ernie"):
                        continue
                    model_path = f"{self.provider_id}:{model_id}"
                    model_info = await get_model_info(model_path)
                    if not model_info:
                        logger.warning(f"Baidu model info not found: {model_path}")
                        continue
                    self.models.append(model_info)
                logger.debug(f"Baidu models loaded: {len(self.models)}")
            except Exception as e:
                logger.error(f"Baidu load models error: {e}")

    def make_reasoning_setting(
        self, model_id: str, reasoning_effort: ReasoningEffortType
    ) -> ModelSettings:
        return ModelSettings()
