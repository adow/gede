# coding=utf-8
#
# MCP 连接服务器配置管理
# 支持从 JSON 文件加载 MCP 服务器配置并管理客户端

#
import json
import os
import logging
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ==================== 配置数据类 ====================


@dataclass
class ServerConfig:
    """单个 MCP 服务器的配置"""

    name: str
    """服务器名称（用于标识）"""

    enable: bool = True
    """是否启用该服务器"""

    # ---- 传输类型 ----
    transport_type: str = "stdio"
    """传输类型：stdio（默认）| sse | streamable-http"""

    # ---- Stdio 传输参数 ----
    command: Optional[str] = None
    """Stdio: 启动命令（如 "npx", "python", "node"）"""

    args: Optional[list[str]] = None
    """Stdio: 命令参数列表"""

    env: Optional[dict[str, str]] = None
    """Stdio: 环境变量字典"""

    # ---- HTTP 传输参数（SSE 和 StreamableHTTP 共用）----
    url: Optional[str] = None
    """SSE/HTTP: 服务器端点 URL"""

    headers: Optional[dict[str, str]] = None
    """SSE/HTTP: HTTP 请求头（如认证信息）"""

    timeout: float = 30.0
    """SSE/HTTP: 连接超时时间（秒）"""

    sse_read_timeout: float = 300.0
    """SSE: 读取超时时间（秒，仅用于 SSE）"""

    terminate_on_close: bool = True
    """StreamableHTTP: 关闭时是否终止会话"""

    # ---- 其他配置 ----
    raw_config: dict[str, Any] = field(default_factory=dict)
    """原始配置字典（用于存储未识别的额外字段）"""

    def validate(self) -> None:
        """验证配置的有效性"""
        if not self.enable:
            return  # 禁用的服务器不需要验证

        # 验证传输类型
        valid_types = ["stdio", "sse", "streamable-http", "streamable_http", "http"]
        if self.transport_type not in valid_types:
            raise ValueError(
                f"服务器 '{self.name}' 的传输类型无效: {self.transport_type}。"
                f"支持的类型: stdio, sse, streamable-http"
            )

        # 验证 Stdio 类型参数
        if self.transport_type == "stdio":
            if not self.command:
                raise ValueError(f"Stdio 服务器 '{self.name}' 缺少 'command' 参数")

        # 验证 SSE/HTTP 类型参数
        elif self.transport_type in [
            "sse",
            "streamable-http",
            "streamable_http",
            "http",
        ]:
            if not self.url:
                raise ValueError(
                    f"{self.transport_type.upper()} 服务器 '{self.name}' 缺少 'url' 参数"
                )

    def expand_paths(self) -> None:
        """展开路径中的环境变量和用户目录（~）"""
        # 展开 args 中的路径
        if self.args:
            self.args = [
                os.path.expanduser(os.path.expandvars(arg)) for arg in self.args
            ]

        # 展开 env 中的路径
        if self.env:
            self.env = {
                k: os.path.expanduser(os.path.expandvars(v))
                for k, v in self.env.items()
            }

    @classmethod
    def from_dict(cls, name: str, config: dict[str, Any]) -> "ServerConfig":
        """
        从字典创建服务器配置

        Args:
            name: 服务器名称
            config: 配置字典

        Returns:
            ServerConfig 实例
        """
        # 确定传输类型
        transport_type = config.get("type", "stdio")

        # 如果没有显式指定 type，但有 command 字段，则为 stdio
        if "type" not in config and "command" in config:
            transport_type = "stdio"

        # 提取已知字段
        server_config = cls(
            name=name,
            enable=config.get("enable", True),
            transport_type=transport_type,
            command=config.get("command"),
            args=config.get("args"),
            env=config.get("env"),
            url=config.get("url"),
            headers=config.get("headers"),
            timeout=config.get("timeout", 30.0),
            sse_read_timeout=config.get("sse_read_timeout", 300.0),
            terminate_on_close=config.get("terminate_on_close", True),
            raw_config=config.copy(),
        )

        # 验证和展开路径
        server_config.validate()
        server_config.expand_paths()

        return server_config


@dataclass
class MCPManager:
    """MCP 客户端管理器 - 管理配置和客户端生命周期"""

    servers: dict[str, ServerConfig] = field(default_factory=dict)
    """所有 MCP 服务器配置，键为服务器名称"""

    config_path: Optional[Path] = None
    """配置文件路径"""

    def get_server(self, name: str) -> Optional[ServerConfig]:
        """
        获取指定名称的服务器配置

        Args:
            name: 服务器名称

        Returns:
            服务器配置，如果不存在或未启用则返回 None
        """
        server = self.servers.get(name)
        if server and server.enable:
            return server
        return None

    def get_enabled_servers(self) -> dict[str, ServerConfig]:
        """
        获取所有启用的服务器配置

        Returns:
            启用的服务器配置字典
        """
        return {name: cfg for name, cfg in self.servers.items() if cfg.enable}

    def list_server_names(self, enabled_only: bool = True) -> list[str]:
        """
        列出所有服务器名称

        Args:
            enabled_only: 是否只列出启用的服务器

        Returns:
            服务器名称列表
        """
        if enabled_only:
            return [name for name, cfg in self.servers.items() if cfg.enable]
        return list(self.servers.keys())

    @classmethod
    def from_file(cls, file_path: str | Path) -> "MCPManager":
        """
        从 JSON 文件加载配置

        Args:
            file_path: JSON 配置文件路径

        Returns:
            MCPManager 实例

        Raises:
            FileNotFoundError: 配置文件不存在
            json.JSONDecodeError: JSON 格式错误
            ValueError: 配置格式不符合要求
        """
        file_path = Path(file_path).expanduser().resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {file_path}")

        logger.info(f"正在加载 MCP 配置文件: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"配置文件 JSON 格式错误: {e.msg}", e.doc, e.pos)

        return cls.from_dict(data, config_path=file_path)

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], config_path: Optional[Path] = None
    ) -> "MCPManager":
        """
        从字典创建管理器

        Args:
            data: 配置字典
            config_path: 配置文件路径（可选）

        Returns:
            MCPManager 实例
        """
        if "mcpServers" not in data:
            raise ValueError(
                "配置文件格式错误: 缺少 'mcpServers' 字段\n"
                "正确格式:\n"
                '{\n  "mcpServers": {\n    "server_name": { ... }\n  }\n}'
            )

        servers_data = data["mcpServers"]
        if not isinstance(servers_data, dict):
            raise ValueError("'mcpServers' 必须是一个字典对象")

        # 解析每个服务器配置
        servers = {}
        for name, server_config in servers_data.items():
            try:
                servers[name] = ServerConfig.from_dict(name, server_config)
                logger.debug(
                    f"✓ 加载服务器配置: {name} ({servers[name].transport_type})"
                )
            except Exception as e:
                logger.warning(f"✗ 跳过服务器 '{name}': {e}")
                continue

        manager = cls(servers=servers, config_path=config_path)

        # 输出统计信息
        enabled_count = len(manager.get_enabled_servers())
        total_count = len(servers)
        logger.info(f"配置加载完成: {enabled_count}/{total_count} 个服务器已启用")

        return manager

    async def create_client(self, server_name: str):
        """
        根据配置创建并连接 MCP 客户端

        Args:
            server_name: 服务器名称

        Returns:
            已连接的 MCP 客户端实例

        Raises:
            ValueError: 服务器不存在或未启用
        """
        from .mcp_client import (
            MCPStdioServer,
            MCPSSEServer,
            MCPHttpServer,
        )

        server = self.get_server(server_name)
        if not server:
            available = self.list_server_names()
            raise ValueError(
                f"服务器 '{server_name}' 不存在或未启用。"
                f"可用服务器: {', '.join(available)}"
            )

        logger.info(f"正在创建客户端: {server_name} ({server.transport_type})")

        # 根据传输类型创建客户端
        if server.transport_type == "stdio":
            if not server.command:
                raise ValueError(f"Stdio 服务器 '{server_name}' 缺少 'command' 参数")
            client = MCPStdioServer()
            await client.connect(
                command=server.command,
                args=server.args,
                env=server.env,
            )
            return client

        elif server.transport_type == "sse":
            if not server.url:
                raise ValueError(f"SSE 服务器 '{server_name}' 缺少 'url' 参数")
            client = MCPSSEServer()
            await client.connect(
                url=server.url,
                headers=server.headers,
                sse_read_timeout=server.sse_read_timeout,
                timeout=server.timeout,
            )
            return client

        elif server.transport_type in ["streamable-http", "streamable_http", "http"]:
            if not server.url:
                raise ValueError(f"HTTP 服务器 '{server_name}' 缺少 'url' 参数")
            client = MCPHttpServer()
            await client.connect(
                url=server.url,
                headers=server.headers,
                timeout=server.timeout,
                terminate_on_close=server.terminate_on_close,
            )
            return client

        else:
            raise ValueError(f"不支持的传输类型: {server.transport_type}")

    async def create_all_clients(self) -> dict[str, Any]:
        """
        创建所有启用服务器的客户端

        Returns:
            客户端字典，键为服务器名称

        Example:
            clients = await manager.create_all_clients()
            try:
                # 使用客户端
                tools = await clients["filesystem"].list_tools()
            finally:
                # 清理
                for client in clients.values():
                    await client.cleanup()
        """
        clients = {}
        enabled_servers = self.get_enabled_servers()

        for name in enabled_servers:
            try:
                clients[name] = await self.create_client(name)
            except Exception as e:
                logger.error(f"创建客户端 '{name}' 失败: {e}")
                # 继续创建其他客户端

        logger.info(f"成功创建 {len(clients)}/{len(enabled_servers)} 个客户端")
        return clients


# ==================== 便捷封装函数 ====================


async def get_mcp_servers(config_path: str = "~/mcp.json"):
    """
    便捷函数：一步加载配置并创建所有MCP服务器

    Args:
        config_path: 配置文件路径，默认为 ~/mcp.json

    Returns:
        tuple[dict, AsyncExitStack]: (服务器字典, AsyncExitStack实例)

    Usage:
        servers, stack = await get_mcp_servers()
        try:
            # 使用服务器
            tools = await servers["filesystem"].list_tools()
        finally:
            await stack.aclose()

    或使用上下文管理器:
        async with await get_mcp_servers() as servers:
            tools = await servers["filesystem"].list_tools()
    """
    from contextlib import AsyncExitStack

    # 展开路径
    config_path = os.path.expanduser(config_path)

    # 加载配置
    manager = MCPManager.from_file(config_path)

    # 创建 AsyncExitStack
    stack = AsyncExitStack()
    servers = {}

    try:
        # 创建所有服务器并加入 stack
        for name in manager.list_server_names():
            try:
                server = await manager.create_client(name)
                await stack.enter_async_context(server)
                servers[name] = server
            except Exception as e:
                logger.error(f"创建服务器 '{name}' 失败: {e}")
                continue

        return servers, stack
    except Exception:
        # 如果出错，清理已创建的资源
        await stack.aclose()
        raise


class MCPServersContext:
    """支持 async with 语法的 MCP 服务器上下文管理器"""

    def __init__(self, config_path: str = "~/mcp.json"):
        self.config_path = config_path
        self.servers = None
        self.stack = None

    async def __aenter__(self):
        self.servers, self.stack = await get_mcp_servers(self.config_path)
        return self.servers

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.stack:
            await self.stack.aclose()
        return False


# ==================== 使用示例 ====================


async def example_usage():
    """示例：从配置文件加载并使用"""
    # 方式 1: 使用上下文管理器（推荐）
    # async with MCPServersContext("./mcp.json") as servers:
    #     for name, server in servers.items():
    #         tools = await server.list_tools()
    #         print(f"{name}: {len(tools)} 个工具")
    #     # 退出时自动清理

    # 方式 2: 手动管理
    servers, stack = await get_mcp_servers("./mcp.json")
    try:
        for name, server in servers.items():
            tools = await server.list_tools()
            print(f"{name}: {len(tools)} 个工具")
    finally:
        await stack.aclose()


async def example_manual_config():
    """示例：手动构建配置"""
    # 方式 1：从字典创建
    config_data = {
        "mcpServers": {
            "my_server": {
                "command": "python",
                "args": ["server.py"],
                "enable": True,
            }
        }
    }
    manager = MCPManager.from_dict(config_data)

    # 方式 2：直接创建 ServerConfig
    server = ServerConfig(
        name="test_server",
        transport_type="streamable-http",
        url="https://api.example.com/mcp",
        headers={"Authorization": "Bearer token"},
        timeout=60.0,
    )
    server.validate()

    manager2 = MCPManager(servers={"test_server": server})
    client = await manager2.create_client("test_server")


if __name__ == "__main__":
    import asyncio

    # 运行示例
    asyncio.run(example_usage())
