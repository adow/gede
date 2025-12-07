# coding=utf-8
#
# base settings
#

from agents import ModelSettings
from openai.types.shared import Reasoning

openrouter_base_settings = ModelSettings(
    extra_headers={
        "X-Title": "gede",
        "HTTP-Referer": "https://gede.slashusr.xyz",
    }
)

openai_gpt_base_settings = ModelSettings(include_usage=True)

openai_gpt_reasoning_settings = ModelSettings(reasoning=Reasoning(effort="medium"))

doubao_base_settings = ModelSettings(
    include_usage=True,
)

grok_web_search_settings = ModelSettings(
    extra_body={"search_parameters": {"mode": "auto"}}
)
