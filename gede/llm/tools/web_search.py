# coding=utf-8

import logging
import os
import sys
import json
from typing import Optional

import asyncio
from pydantic import BaseModel
from rich.panel import Panel
import httpx
from agents import function_tool, RunContextWrapper
from ...commands import CommandConext


from ...top import (
    logger,
)


class WebSearchResult(BaseModel):
    title: Optional[str] = None

    url: str

    # Summary
    summary: Optional[str] = None

    # Original content
    content: Optional[str] = None

    # Text that matches search requirements
    answer: Optional[str] = None

    # Index
    index: Optional[int] = None

    # Source
    source: Optional[str] = None

    # Domain
    domain: Optional[str] = None

    # Similarity
    similarity: Optional[float] = None

    # Confidence
    confidence: Optional[float] = None

    # Published date
    published_date: Optional[str] = None

    # Icon
    site_icon: Optional[str] = None


async def exa_web_search(
    query: str,
    summary_query: Optional[str] = None,
    limit: int = 10,
):
    """
    Use exa to search the internet
    Args:
        query (str): Query keywords
        summary_query (str): Requirements for content summarization (does not return full text), if empty, no summarization is performed, and full content is returned
        limit (int): Limit the number of search results, default is 10
        read_content (bool): Whether to read content, default is True
    Returns:
        List of search results
    """
    EXAAI_API_KEY = os.getenv("EXAAI_API_KEY", "")
    EXAAI_BASE_URL = os.getenv("EXAAI_BASE_URL", "https://api.exa.ai")
    url = f"{EXAAI_BASE_URL}/search"
    logger.debug("url: %s", url)
    headers = {
        "Authorization": f"Bearer {EXAAI_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "query": query,
        "contents": {
            "text": not summary_query,
        },
        "useAutoprompt": True,
        "numResults": limit,
        "type": "auto",
    }

    if summary_query:
        data["contents"]["summary"] = {
            "query": summary_query,
        }
    logger.debug(json.dumps(headers, indent=2))
    logger.debug(json.dumps(data, indent=2))
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=data, timeout=180)
            if response.status_code != 200:
                logger.error(
                    f"Failed to search {query} from exa, {response.status_code}, {response.text}"
                )
                return None
            result = response.json()
            results = result.get("results", [])
            if not results:
                logger.error(f"No results found for {query}")
                return None
        except httpx.RequestError as e:
            logger.error(f"Request error while searching {query} from exa: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error while searching {query} from exa: {e}")
            return None
    search_results: list[WebSearchResult] = []
    for pos, one_result in enumerate(results):
        title = one_result.get("title")
        url = one_result.get("url")
        description = one_result.get("summary")
        content = one_result.get("text")
        published_date = one_result.get("publishedDate")
        # logger.debug(f"[{pos + 1}] {title} {url}")
        # logger.debug(description)
        # logger.debug(content)

        web_search_result = WebSearchResult(
            title=title,
            url=url,
            summary=description,
            answer=description,
            content=content,
            index=pos + 1,
            published_date=published_date,
        )

        search_results.append(web_search_result)

    return search_results


@function_tool(name_override="web_search_summary_tool")
async def web_search_summary_tool(
    wrapper: RunContextWrapper[CommandConext], query: str, summary_query: str
):
    """
    Perform internet search, use this tool when the user's question requires the latest information from the internet.
    After obtaining search results, each content will be summarized.
    Args:
        query (str): Query keyword combination
        summary_query (str): Requirements for content summarization
    """
    logger.debug("web_search_tool query: %s, summary: %s", query, summary_query)
    context = wrapper.context
    # context.print_tool_info(
    #     f"web_search_summary_tool query: {query}, summary_query: {summary_query}"
    # )

    results = await exa_web_search(query=query, summary_query=summary_query)
    return results


@function_tool(name_override="web_search_full_tool")
async def web_search_full_tool(warpper: RunContextWrapper[CommandConext], query: str):
    """
    Perform internet search, use this tool when the user's question requires the latest information from the internet.
    When search results are obtained, the full content of the corresponding links will also be retrieved.
    Args:
        query (str): Query keyword combination
    """
    logger.debug("web_search_tool query: %s", query)
    context = warpper.context
    # context.print_tool_info(f"web_search_full_tool query: {query}")
    results = await exa_web_search(query=query)
    return results


# test
async def test_exa_web_search():
    query = "About Leica M3"
    results = await exa_web_search(query=query)
    if not results:
        print("No results found")
        return
    for res in results:
        print(f"Title: {res.title}")
        print(f"URL: {res.url}")
        print(f"Published Date: {res.published_date}")
        print(f"Summary: {res.summary}")
        print("Content")
        print(res.content)
        print("-" * 80)


async def tests():
    await test_exa_web_search()


if __name__ == "__main__":
    from ...config import load_config

    logger.setLevel(logging.DEBUG)
    load_config()
    asyncio.run(tests())
