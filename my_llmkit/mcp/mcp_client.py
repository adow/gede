# coding=utf-8
#
# 连接 MCP Server 实现
# 支持三种传输方式：Stdio、SSE、StreamableHTTP
#

import logging
from abc import ABC, abstractmethod
from typing import Optional, Any
from contextlib import AsyncExitStack
from datetime import timedelta

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool, CallToolResult

logger = logging.getLogger(__name__)


# ==================== MCP Client 基类 ====================


class MCPServerBase(ABC):
    """MCP 客户端基类，提供通用的会话管理和工具调用接口"""

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self._tools_cache: Optional[list[Tool]] = None

    @abstractmethod
    async def connect(self, *args, **kwargs):
        """连接到 MCP Server，子类必须实现此方法"""
        pass

    async def initialize(self):
        """初始化 MCP 会话"""
        if not self.session:
            raise RuntimeError("Session not created, call connect() first")

        try:
            result = await self.session.initialize()
            logger.info(
                f"✓ MCP 连接成功: {result.serverInfo.name} "
                f"(protocol: {result.protocolVersion})"
            )
            return result
        except Exception as e:
            logger.error(f"✗ MCP 初始化失败: {e}")
            raise

    async def list_tools(self, force_refresh: bool = False) -> list[Tool]:
        """
        列出所有可用工具

        Args:
            force_refresh: 是否强制刷新缓存

        Returns:
            工具列表
        """
        if not self.session:
            raise RuntimeError("未连接到 MCP Server")

        # 如果有缓存且不强制刷新，返回缓存
        if self._tools_cache and not force_refresh:
            return self._tools_cache

        try:
            result = await self.session.list_tools()
            self._tools_cache = result.tools
            logger.info(f"可用工具: {[tool.name for tool in self._tools_cache]}")
            return self._tools_cache
        except Exception as e:
            logger.error(f"获取工具列表失败: {e}")
            raise

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: timedelta | None = None,
    ) -> CallToolResult:
        """
        调用指定的工具

        Args:
            name: 工具名称
            arguments: 工具参数
            read_timeout_seconds: 读取超时时间（秒）

        Returns:
            工具调用结果
        """
        if not self.session:
            raise RuntimeError("未连接到 MCP Server")

        try:
            logger.debug(f"调用 MCP 工具: {name}, 参数: {arguments}")
            result = await self.session.call_tool(
                name=name,
                arguments=arguments or {},
                read_timeout_seconds=read_timeout_seconds,
            )

            if result.isError:
                logger.warning(f"工具 {name} 返回错误: {result.content}")
            else:
                logger.debug(f"工具 {name} 执行成功")

            return result
        except Exception as e:
            logger.error(f"调用工具 {name} 失败: {e}")
            raise

    async def cleanup(self):
        """清理资源，关闭所有连接"""
        try:
            await self.exit_stack.aclose()
            self.session = None
            self._tools_cache = None
            logger.info("MCP 连接已关闭")
        except Exception as e:
            logger.error(f"清理资源时出错: {e}")

    async def __aenter__(self):
        """支持 async with 语句"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """自动清理资源"""
        await self.cleanup()


# ==================== Stdio Transport 实现 ====================


class MCPStdioServer(MCPServerBase):
    """
    基于 Stdio 的 MCP 服务器连接
    用于本地进程通信（Python/Node.js 服务器）
    """

    async def connect(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ):
        """
        连接到本地 MCP 服务器进程

        Args:
            command: 启动命令（如 "python" 或 "node"）
            args: 命令参数列表（如 ["server.py"]）
            env: 环境变量字典
        """
        try:
            server_params = StdioServerParameters(
                command=command,
                args=args or [],
                env=env,
            )

            logger.info(
                f"正在连接到 Stdio MCP Server: {command} {' '.join(args or [])}"
            )

            # 使用 exit_stack 管理 stdio_client 的生命周期
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read_stream, write_stream = stdio_transport

            # 创建 ClientSession
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            # 初始化会话
            await self.initialize()

        except Exception as e:
            logger.error(f"连接 Stdio MCP Server 失败: {e}")
            await self.cleanup()
            raise

    @classmethod
    async def connect_python_server(
        cls, script_path: str, env: dict[str, str] | None = None
    ):
        """
        便捷方法：连接到 Python MCP 服务器

        Args:
            script_path: Python 脚本路径
            env: 环境变量

        Returns:
            已连接的客户端实例
        """
        client = cls()
        await client.connect(command="python", args=[script_path], env=env)
        return client

    @classmethod
    async def connect_node_server(
        cls, script_path: str, env: dict[str, str] | None = None
    ):
        """
        便捷方法：连接到 Node.js MCP 服务器

        Args:
            script_path: JavaScript 脚本路径
            env: 环境变量

        Returns:
            已连接的客户端实例
        """
        client = cls()
        await client.connect(command="node", args=[script_path], env=env)
        return client


# ==================== SSE Transport 实现 ====================


class MCPSSEServer(MCPServerBase):
    """
    基于 SSE (Server-Sent Events) 的 MCP 服务器连接
    用于 HTTP SSE 长连接
    """

    async def connect(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        sse_read_timeout: float = 300.0,
        timeout: float = 5.0,
    ):
        """
        连接到 SSE MCP 服务器

        Args:
            url: SSE 端点 URL
            headers: HTTP 请求头（如认证信息）
            sse_read_timeout: SSE 读取超时（秒，默认 5 分钟）
            timeout: 连接超时（秒）
        """
        try:
            from mcp.client.sse import sse_client

            logger.info(f"正在连接到 SSE MCP Server: {url}")

            # 使用 exit_stack 管理 sse_client 的生命周期
            sse_transport = await self.exit_stack.enter_async_context(
                sse_client(
                    url=url,
                    headers=headers,
                    sse_read_timeout=sse_read_timeout,
                    timeout=timeout,
                )
            )
            read_stream, write_stream = sse_transport

            # 创建 ClientSession
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            # 初始化会话
            await self.initialize()

        except Exception as e:
            logger.error(f"连接 SSE MCP Server 失败: {e}")
            await self.cleanup()
            raise

    @classmethod
    async def connect_with_bearer_token(
        cls,
        url: str,
        token: str,
        sse_read_timeout: float = 300.0,
    ):
        """
        便捷方法：使用 Bearer Token 认证连接

        Args:
            url: SSE 端点 URL
            token: Bearer Token
            sse_read_timeout: SSE 读取超时

        Returns:
            已连接的客户端实例
        """
        client = cls()
        headers = {"Authorization": f"Bearer {token}"}
        await client.connect(
            url=url, headers=headers, sse_read_timeout=sse_read_timeout
        )
        return client


# ==================== StreamableHTTP Transport 实现 ====================


class MCPHttpServer(MCPServerBase):
    """
    基于 StreamableHTTP 的 MCP 服务器连接
    推荐用于生产环境的 HTTP 传输
    """

    def __init__(self):
        super().__init__()
        self.session_id: Optional[str] = None

    async def connect(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        terminate_on_close: bool = True,
    ):
        """
        连接到 StreamableHTTP MCP 服务器

        Args:
            url: HTTP 端点 URL
            headers: HTTP 请求头（如认证信息）
            timeout: 连接超时（秒）
            terminate_on_close: 关闭时是否终止会话
        """
        try:
            from mcp.client.streamable_http import streamable_http_client
            import httpx

            logger.info(f"正在连接到 StreamableHTTP MCP Server: {url}")

            # 创建 httpx 客户端
            http_client = httpx.AsyncClient(
                headers=headers or {},
                timeout=httpx.Timeout(timeout),
            )

            # 将 http_client 添加到 exit_stack 管理，确保自动清理
            await self.exit_stack.enter_async_context(http_client)

            # 使用 exit_stack 管理 streamable_http_client 的生命周期
            http_transport = await self.exit_stack.enter_async_context(
                streamable_http_client(
                    url=url,
                    http_client=http_client,
                    terminate_on_close=terminate_on_close,
                )
            )
            read_stream, write_stream, get_session_id = http_transport

            # 创建 ClientSession
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            # 初始化会话
            await self.initialize()

            # 获取会话 ID（用于重连或调试）
            self.session_id = get_session_id()
            logger.info(f"会话 ID: {self.session_id}")

        except Exception as e:
            logger.error(f"连接 StreamableHTTP MCP Server 失败: {e}")
            await self.cleanup()
            raise

    @classmethod
    async def connect_with_bearer_token(
        cls,
        url: str,
        token: str,
        timeout: float = 30.0,
    ):
        """
        便捷方法：使用 Bearer Token 认证连接

        Args:
            url: HTTP 端点 URL
            token: Bearer Token
            timeout: 连接超时

        Returns:
            已连接的客户端实例
        """
        client = cls()
        headers = {"Authorization": f"Bearer {token}"}
        await client.connect(url=url, headers=headers, timeout=timeout)
        return client


MCPServerType = MCPStdioServer | MCPSSEServer | MCPHttpServer
