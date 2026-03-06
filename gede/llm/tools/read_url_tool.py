# coding=utf-8
#
#

import logging
import httpx

logger = logging.getLogger(__name__)


async def read_url(url: str):
    """
    Read webpage content, use this tool when you need to get information from a specified URL
    Args:
        url: The URL of the webpage to read
    Returns:
        Returns the text content of the webpage, or an error message if unable to read
    """
    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            response = await client.get(url, timeout=60)
            if response.status_code != 200:
                logger.error(
                    f"Failed to read {url}, status code: {response.status_code}"
                )
                return f"Unable to read URL: {url}"
            return response.text
        except httpx.RequestError as e:
            logger.error(f"Request error while reading {url}  {e}")
            return f"Request error while reading {url} : {e}"
        except Exception as e:
            logger.error(f"Unexpected error while reading {url} : {e}")
            return f"Unexpected error while reading {url} : {e}"
