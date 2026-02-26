# coding=utf-8
#
# config.py
#
# load env file
#

import os
import logging
from dotenv import load_dotenv

from .top import gede_config_dir

logger = logging.getLogger(__name__)


def get_config_filepath():
    config_dir = gede_config_dir()
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

# zenmux
ZENMUX_API_KEY=""
ZENMUX_BASE_URL_OPENAI="https://zenmux.ai/api/v1"
ZENMUX_BASE_URL_ANTHROPIC="https://zenmux.ai/api/anthropic"

OPENAI_API_KEY=""
OPENAI_BASE_URL="https://api.openai.com/v1"

# google gemini (OpenAI compatible)
GEMINI_API_KEY=""
GEMINI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"

# baidu 
QIANFAN_API_KEY=''
QIANFAN_BASE_URL='https://qianfan.baidubce.com/v2'

# SiliconFlow
SILICONFLOW_API_KEY=""
SILICONFLOW_BASE_URL="https://api.siliconflow.cn/v1"

# aliyun qwen
DASHSCOPE_API_KEY=""
DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"

# doubao
ARK_API_KEY=""
ARK_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"

# deepseek
DEEPSEEK_API_KEY=""
DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"

# jina
JINA_API_KEY=""
JINA_BASE_URL="https://r.jina.ai"

# exa.ai
EXAAI_API_KEY=""
EXAAI_BASE_URL="https://api.exa.ai"

# bocha
BOCHA_API_KEY=""
BOCHA_BASE_URL="https://api.302ai.cn/bochaai/v1"


# generate title
GENERATE_TITLE_MODEL=""

# phoenix
PHOENIX_API_KEY=""
PHOENIX_CLIENT_HEADERS="api_key=${PHOENIX_API_KEY}"
PHOENIX_COLLECTOR_ENDPOINT="https://app.phoenix.arize.com"
# Phoenix trace endpoint (used when --trace flag is enabled with arize-trace extension installed)
# PHOENIX_COLLECTOR_ENDPOINT="https://app.phoenix.arize.com/s/your-project-token/v1/traces"

# DEBUG=true
# SDK log level defaults (effective when --log-level is not provided):
# OPENAI_LOG="debug"      # allowed: "debug" / "info"
# ANTHROPIC_LOG="debug"   # allowed: "debug" / "info"
            """
        f.write(default_content.strip())
        logger.info(f"Default .env file created at {env_filename}")
        logger.warning("Please edit it to add your API keys.")


load_config()
