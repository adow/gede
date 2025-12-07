# coding=utf-8

import os
from typing import Dict, List, Optional, Union, Literal, Any
from dataclasses import dataclass
from pathlib import Path
import json
from ...top import logger
from ...config import get_config_dir


# Server type definitions
ServerType = Literal["stdio", "sse", "streamable-http"]


@dataclass
class StdioServerConfig:
    """STDIO type server configuration"""

    command: str
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None
    auto_select: bool = False
    enable: bool = True


@dataclass
class RemoteServerConfig:
    """Remote server configuration (SSE or Streamable HTTP)"""

    type: Literal["sse", "streamable-http"]
    url: str
    headers: Optional[Dict[str, str]] = None
    note: Optional[str] = None
    auto_select: bool = False
    enable: bool = True


# Union type representing any server configuration
AnyServerConfig = Union[StdioServerConfig, RemoteServerConfig]


@dataclass
class ParsedServerConfig:
    """Parsed server configuration"""

    name: str
    type: ServerType
    config: Union[StdioServerConfig, RemoteServerConfig]


class MCPConfigReader:
    """MCP configuration reader"""

    def __init__(self) -> None:
        self._config: Optional[Dict[str, Any]] = None

    def create_default_config(self, file_path: Union[str, Path]) -> None:
        """
        Create default configuration file
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"mcpServers": {}}
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"Default configuration file created: {file_path}")

    def load_from_file(self, file_path: Union[str, Path]) -> None:
        """Load configuration from file

        Args:
            file_path: Configuration file path

        Raises:
            FileNotFoundError: File does not exist
            json.JSONDecodeError: JSON format error
            ValueError: Configuration format error
        """
        try:
            path = Path(file_path)
            content = path.read_text(encoding="utf-8")
            self.load_from_string(content)
        except FileNotFoundError:
            logger.warning(f"Configuration file does not exist: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to read configuration file: {e}")

    def load_from_string(self, content: str) -> None:
        """Load configuration from string

        Args:
            content: JSON string content

        Raises:
            json.JSONDecodeError: JSON format error
            ValueError: Configuration format error
        """
        try:
            config = json.loads(content)
            self._validate_config(config)
            self._config = config
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"JSON format error: {e.msg}", e.doc, e.pos)
        except Exception as e:
            raise ValueError(f"Configuration parsing failed: {e}")

    def _validate_config(self, config: Any) -> None:
        """Validate configuration format"""
        if not isinstance(config, dict):
            raise ValueError("Configuration must be an object")

        if "mcpServers" not in config:
            raise ValueError("Configuration must contain mcpServers field")

        if not isinstance(config["mcpServers"], dict):
            raise ValueError("mcpServers must be an object")

        # Validate each server configuration
        for name, server_config in config["mcpServers"].items():
            self._validate_server_config(name, server_config)

    def _validate_server_config(self, name: str, config: Any) -> None:
        """Validate individual server configuration"""
        if not isinstance(config, dict):
            raise ValueError(f"Server '{name}' configuration must be an object")

        server_type = self._get_server_type(config)

        if server_type == "stdio":
            if "command" not in config or not isinstance(config["command"], str):
                raise ValueError(f"STDIO server '{name}' must have command field")

            if "args" in config and not isinstance(config["args"], list):
                raise ValueError(f"STDIO server '{name}' args must be an array")

            if "env" in config and not isinstance(config["env"], dict):
                raise ValueError(f"STDIO server '{name}' env must be an object")

        elif server_type in ["sse", "streamable-http"]:
            if "url" not in config or not isinstance(config["url"], str):
                raise ValueError(
                    f"{server_type.upper()} server '{name}' must have url field"
                )

    def _get_server_type(self, config: Dict[str, Any]) -> ServerType:
        """Determine server type"""
        if "type" in config:
            if config["type"] in ["sse", "streamable-http"]:
                return config["type"]
            else:
                raise ValueError(f"Unsupported server type: {config['type']}")

        # If no type field but has command, consider it stdio
        if "command" in config:
            return "stdio"

        raise ValueError("Unable to determine server type")

    def _parse_server_config(
        self, name: str, raw_config: Dict[str, Any]
    ) -> ParsedServerConfig:
        """Parse server configuration"""
        server_type = self._get_server_type(raw_config)

        if server_type == "stdio":
            config = StdioServerConfig(
                command=raw_config["command"],
                args=raw_config.get("args"),
                env=raw_config.get("env"),
                cwd=raw_config.get("cwd"),
                auto_select=raw_config.get("auto_select", False),
                enable=raw_config.get("enable", True),
            )
        else:  # sse or streamable-http
            config = RemoteServerConfig(
                type=server_type,
                url=raw_config["url"],
                headers=raw_config.get("headers"),
                note=raw_config.get("note"),
                auto_select=raw_config.get("auto_select", False),
                enable=raw_config.get("enable", True),
            )

        return ParsedServerConfig(name=name, type=server_type, config=config)

    def get_all_servers(self) -> List[ParsedServerConfig]:
        """Get all server configurations"""
        if not self._config:
            logger.warning("Configuration not loaded")
            return []

        servers = []
        for name, raw_config in self._config["mcpServers"].items():
            servers.append(self._parse_server_config(name, raw_config))

        return servers

    def get_servers_by_type(self, server_type: ServerType) -> List[ParsedServerConfig]:
        """Get server configurations by type"""
        return [
            server for server in self.get_all_servers() if server.type == server_type
        ]

    def get_server(self, name: str) -> Optional[ParsedServerConfig]:
        """Get specific server configuration"""
        if not self._config or name not in self._config["mcpServers"]:
            return None

        raw_config = self._config["mcpServers"][name]
        return self._parse_server_config(name, raw_config)

    def get_stdio_servers(self) -> List[Dict[str, Any]]:
        """Get STDIO type server configurations"""
        stdio_servers = []
        for server in self.get_servers_by_type("stdio"):
            if isinstance(server.config, StdioServerConfig):
                stdio_servers.append(
                    {
                        "name": server.name,
                        "command": server.config.command,
                        "args": server.config.args,
                        "env": server.config.env,
                    }
                )
        return stdio_servers

    def get_sse_servers(self) -> List[Dict[str, Any]]:
        """Get SSE type server configurations"""
        sse_servers = []
        for server in self.get_servers_by_type("sse"):
            if isinstance(server.config, RemoteServerConfig):
                sse_servers.append(
                    {
                        "name": server.name,
                        "url": server.config.url,
                        "headers": server.config.headers,
                        "note": server.config.note,
                    }
                )
        return sse_servers

    def get_streamable_http_servers(self) -> List[Dict[str, Any]]:
        """Get Streamable HTTP type server configurations"""
        http_servers = []
        for server in self.get_servers_by_type("streamable-http"):
            if isinstance(server.config, RemoteServerConfig):
                http_servers.append(
                    {
                        "name": server.name,
                        "url": server.config.url,
                        "headers": server.config.headers,
                        "note": server.config.note,
                    }
                )
        return http_servers

    def is_loaded(self) -> bool:
        """Check if configuration is loaded"""
        return self._config is not None

    def get_raw_config(self) -> Optional[Dict[str, Any]]:
        """Get raw configuration object"""
        return self._config


def load_mcp_config(file_path: Union[str, Path]) -> MCPConfigReader:
    """Load MCP configuration

    Args:
        file_path: Configuration file path

    Returns:
        MCPConfigReader instance

    Raises:
        FileNotFoundError: File does not exist
        json.JSONDecodeError: JSON format error
        ValueError: Configuration format error
    """
    reader = MCPConfigReader()
    if not Path(file_path).exists():
        reader.create_default_config(file_path)
    reader.load_from_file(file_path)
    return reader


# test


def test_load_mcp_config():
    """Test MCP configuration loading"""
    try:
        config_dir = get_config_dir()
        filename = os.path.join(config_dir, "mcp.json")
        config_reader = load_mcp_config(filename)
        print("MCP configuration loaded successfully")
        print("All servers:", config_reader.get_all_servers())
        print("STDIO servers:", config_reader.get_stdio_servers())
        print("SSE servers:", config_reader.get_sse_servers())
        print("Streamable HTTP servers:", config_reader.get_streamable_http_servers())
    except Exception as e:
        print(f"Failed to load MCP configuration: {e}")


if __name__ == "__main__":
    # Test MCP configuration loading
    test_load_mcp_config()
