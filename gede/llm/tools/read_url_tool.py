# coding=utf-8
#
#
from datetime import datetime, timezone

import httpx
from agents import function_tool, RunContextWrapper
from ...top import logger
from rich.panel import Panel
from ...commands import CommandConext


async def read_url(url: str):
    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            response = await client.get(url, timeout=60)
            if response.status_code != 200:
                logger.error(
                    f"Failed to read {url}, status code: {response.status_code}"
                )
                return None
            return response.text
        except httpx.RequestError as e:
            logger.error(f"Request error while reading {url}  {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error while reading {url} : {e}")
            return None


@function_tool(name_override="read_page_tool")
async def read_page_tool(wrapper: RunContextWrapper[CommandConext], url: str) -> str:
    """
    Read webpage content, use this tool when you need to get information from a specified URL
    Args:
        url: The URL of the webpage to read
    Returns:
        Returns the text content of the webpage, or an error message if unable to read
    """
    context = wrapper.context
    context.print_tool_info(f"read_page_tool: {url}")

    content = await read_url(url)
    if content is None:
        return f"Unable to read URL: {url}"

    return content
