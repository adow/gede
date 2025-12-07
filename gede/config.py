# coding=utf-8
#
# config.py
#
# load env file
#

import os
from pathlib import Path
from dotenv import load_dotenv

from .top import logger


def get_config_dir():
    config_dir = os.path.join(Path.home(), ".gede", "config")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    return config_dir


def get_config_filepath():
    config_dir = get_config_dir()
    return os.path.join(config_dir, ".env")


def load_config():
    env_filename = get_config_filepath()
    if not os.path.exists(env_filename):
        create_default_env()
    load_dotenv(env_filename)


def create_default_env():
    env_filename = get_config_filepath()
    with open(env_filename, "w") as f:
        default_content = """
# 302.ai
AI302_API_KEY=""
AI302_BASE_URL="https://api.302ai.cn/v1"

# openrouter.ai
OPENROUTER_API_KEY=""
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"

OPENAI_API_KEY=""
OPENAI_BASE_URL="https://api.openai.com/v1"

# v2 version
WENXIN_API_KEY=''
WENXIN_BASE_URL='https://qianfan.baidubce.com/v2'

# SiliconFlow
SILICONFLOW_API_KEY=""
SILICONFLOW_BASE_URL="https://api.siliconflow.cn/v1"

# jina
JINA_API_KEY=""
JINA_BASE_URL="https://r.jina.ai"

# exa.ai
EXAAI_API_KEY=""
EXAAI_BASE_URL="https://api.exa.ai"

# bocha
BOCHA_API_KEY=""
BOCHA_BASE_URL="https://api.302ai.cn/bochaai/v1"

# aliyun qwen
QWEN_API_KEY=""
QWEN_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"

# doubao
DOUBAO_API_KEY=""
DOUBAO_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
DOUBAO_BOT_URL="https://ark.cn-beijing.volces.com/api/v3/bots"

# deepseek
DEEPSEEK_API_KEY=""
DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"

# hunyuan
HY_APPID=""
HY_SECRETID=""
HY_SECRETKEY=""
HY_API_KEY=""
HY_BASE_URL="https://api.hunyuan.cloud.tencent.com/v1"

# vllm
VLLM_API_KEY=""
# qwen3-8b
VLLM_QWEN3_BASE_URL=""
VLLM_QWEN3_MODEL="Qwen/Qwen3-8B"

# generate title
GENERATE_TITLE_MODEL=""

# phoenix
PHOENIX_API_KEY=""
PHOENIX_CLIENT_HEADERS="api_key=${PHOENIX_API_KEY}"
PHOENIX_COLLECTOR_ENDPOINT="https://app.phoenix.arize.com"
# Phoenix trace endpoint (used when --trace flag is enabled with arize-trace extension installed)
# PHOENIX_COLLECTOR_ENDPOINT="https://app.phoenix.arize.com/s/your-project-token/v1/traces"

# DEBUG=true
# OPENAI_LOG="debug"
            """
        f.write(default_content.strip())
        logger.info(f"Default .env file created at {env_filename}")
        logger.warning("Please edit it to add your API keys.")


load_config()
