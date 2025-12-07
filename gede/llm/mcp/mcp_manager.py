# coding=utf-8

from dataclasses import dataclass
import os
import sys
from typing import Optional, Union, Any

import asyncio
from contextlib import ExitStack, AsyncExitStack

from .mcp_config import (
    RemoteServerConfig,
    StdioServerConfig,
    load_mcp_config,
)
from agents.mcp import (
    MCPServer,
    MCPServerStdio,
    MCPServerStdioParams,
    MCPServerSse,
    MCPServerSseParams,
    MCPServerStreamableHttp,
    MCPServerStreamableHttpParams,
)
from rich.console import Console
from ...top import logger
from ...config import get_config_dir


@dataclass
class MCPServerItem:
    server: MCPServer
    auto_select: bool = False
    selected: bool = False
    # Only connect once, used to mark if already connected
    connected: bool = False

    async def select_server(self, console: Console):
        self.selected = True
        console.print(f"Select MCP: {self.server.name}", style="info")
        tools = await self.server.list_tools()
        for tool in tools:
            console.print(f"- Tool: {tool.name}", style="info")

    def unselect_server(self, console: Console):
        self.selected = False
        console.print(f"Unselect MCP: {self.server.name}", style="info")


class MCPServerManager:
    """
    Manage MCP server connections
    """

    server_items: dict[str, MCPServerItem] = {}

    def add_server(self, server: MCPServer, auto_select: bool = False):
        self.server_items[server.name] = MCPServerItem(
            server=server, auto_select=auto_select
        )

    def get_server(self, name: str) -> Optional[MCPServerItem]:
        return self.server_items.get(name)

    def get_running_servers(self) -> list[MCPServer]:
        return [item.server for item in self.server_items.values() if item.selected]

    @classmethod
    def get_manager_from_config(cls):
        manager = cls()
        config_dir = get_config_dir()
        config_file = os.path.join(config_dir, "mcp.json")
        configure = load_mcp_config(config_file)
        server_config_list = configure.get_all_servers()
        logger.debug("server config list: %s", server_config_list)
        for one_server_config in server_config_list:
            if not one_server_config.config.enable:
                logger.debug(f'server "{one_server_config.name}" is disabled, skip it')
                continue
            server: Optional[MCPServer] = None
            if one_server_config.type == "stdio" and isinstance(
                one_server_config.config, StdioServerConfig
            ):
                config: StdioServerConfig = one_server_config.config
                params = MCPServerStdioParams(command=config.command)
                if config.args:
                    params["args"] = config.args
                if config.env:
                    params["env"] = config.env
                if config.cwd:
                    params["cwd"] = config.cwd
                server = MCPServerStdio(
                    name=one_server_config.name,
                    params=params,
                    client_session_timeout_seconds=30,
                    cache_tools_list=True,
                )

            elif one_server_config.type == "sse" and isinstance(
                one_server_config.config, RemoteServerConfig
            ):
                sse_config: RemoteServerConfig = one_server_config.config
                params = MCPServerSseParams(url=sse_config.url, timeout=60)
                if sse_config.headers:
                    params["headers"] = sse_config.headers
                server = MCPServerSse(
                    name=one_server_config.name,
                    params=params,
                    cache_tools_list=True,
                )

            elif one_server_config.type == "streamable-http" and isinstance(
                one_server_config.config, RemoteServerConfig
            ):
                streamable_http_config: RemoteServerConfig = one_server_config.config
                params = MCPServerStreamableHttpParams(
                    url=streamable_http_config.url, timeout=60
                )
                if streamable_http_config.headers:
                    params["headers"] = streamable_http_config.headers
                server = MCPServerStreamableHttp(
                    name=one_server_config.name, params=params, cache_tools_list=True
                )
            else:
                logger.warning(
                    "unknown server type: %s, %s",
                    one_server_config.type,
                    one_server_config.name,
                )
                continue

            if server:
                manager.add_server(
                    server, auto_select=one_server_config.config.auto_select
                )

        return manager


# test
async def test_connect_server():
    manager = MCPServerManager.get_manager_from_config()
    async with AsyncExitStack() as stack:
        for item in manager.server_items.values():
            await stack.enter_async_context(item.server)
            await item.server.connect()
            item.selected = True
            logger.info(f'MCP server "{item.server.name}" connected')

        running_servers = manager.get_running_servers()
        print(f"Running servers: {[s.name for s in running_servers]}")
        for server in running_servers:
            tools = await server.list_tools()
            for one in tools:
                logger.debug(f'Tool "{one.name}": {one.description}')
                logger.info(f"Tool: {server.name}/{one.name}")


if __name__ == "__main__":
    asyncio.run(test_connect_server())
