# coding=utf-8
#
# my_llmkit 包的日志配置
# 使用标准 StreamHandler，不依赖第三方库
#

import logging

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger("my_llmkit")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    logger.addHandler(handler)
