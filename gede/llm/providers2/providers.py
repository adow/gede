# coding=utf-8
#
# providers
#

import os
import logging
from pydantic import BaseModel, TypeAdapter

from my_llmkit.models import get_model_info

from ...top import logger, gede_data_dir
from .base import LLMProviderBase
from .openrouter import OpenRouterProvider
from .zenmux import ZenMuxProvider

PROVIDERS: list[LLMProviderBase] = [
    OpenRouterProvider(),
    ZenMuxProvider(),
]


# cache
#
class ModelCache(BaseModel):
    model_id: str
    name: str
    model_path: str

    @property
    async def model_info(self):
        return await get_model_info(self.model_path)


class ProviderCache(BaseModel):
    provider_id: str
    name: str
    models: list[ModelCache] = []


# 启用模型列表
MODEL_DATA: list[ProviderCache] = []
ProviderCacheListType = TypeAdapter(list[ProviderCache])


def load_models_from_file():
    """
    从文件读取模型数据到全局变量 MODEL_DATA
    """
    global MODEL_DATA
    filename = os.path.join(gede_data_dir(), "models.json")
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
            MODEL_DATA = ProviderCacheListType.validate_json(content)


def save_models_to_file():
    """
    保存全局变量 MODEL_DATA 到文件
    """
    filename = os.path.join(gede_data_dir(), "models.json")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(ProviderCacheListType.dump_json(MODEL_DATA, indent=2).decode("utf-8"))
        logger.debug("Models cache saved.")


async def prepare_models():
    """
    启动的时候准备模型列表缓存
    """
    global MODEL_DATA
    if MODEL_DATA:
        return MODEL_DATA
    load_models_from_file()
    if MODEL_DATA:
        return MODEL_DATA
    for provider in PROVIDERS:
        provider_cache = ProviderCache(
            provider_id=provider.provider_id, name=provider.name, models=[]
        )
        for model_id in provider.default_models or []:
            model_path = f"{provider.provider_id}:{model_id}"
            model_info = await get_model_info(model_path)
            if not model_info:
                logger.warning(f"Model {model_path} info not found, skipping.")
                continue
            provider_cache.models.append(
                ModelCache(
                    model_id=model_id,
                    name=model_info.model_name or model_id,
                    model_path=model_path,
                )
            )
        MODEL_DATA.append(provider_cache)
    save_models_to_file()
    return MODEL_DATA


async def add_model(provider_id: str, model_id: str):
    global MODEL_DATA
    for provider in MODEL_DATA:
        if provider.provider_id == provider_id:
            # check model exists
            find_model = [one for one in provider.models if one.model_id == model_id]
            if not find_model:
                model_path = f"{provider_id}:{model_id}"
                model_info = await get_model_info(model_path)
                if not model_info:
                    logger.warning(f"Model {model_path} info not found, cannot add.")
                    return
                provider.models.append(
                    ModelCache(
                        model_id=model_id,
                        name=model_info.model_name or model_id,
                        model_path=model_path,
                    )
                )
            save_models_to_file()
            break


async def remove_model(provider_id: str, model_id: str):
    global MODEL_DATA
    for provider in MODEL_DATA:
        if provider.provider_id == provider_id:
            # check model exists and remove
            provider.models = [
                one for one in provider.models if one.model_id != model_id
            ]
            save_models_to_file()
            break


# tests


async def tests():
    print(await prepare_models())
    for provider in MODEL_DATA:
        for model in provider.models:
            info = await model.model_info
            print(f"{model.model_path} -> {info}")


if __name__ == "__main__":
    import asyncio

    logger.setLevel(logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    asyncio.run(tests())
