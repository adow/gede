# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Dependencies

```bash
# Install dependencies and sync environment
uv sync

# Activate virtual environment (if needed)
source .venv/bin/activate
```

### Running the Application

```bash
# Run the CLI application
uv run gede

# Alternative method
python3 -m gede.gede
```

### Building and Distribution

```bash
# Build the package
uv build

# Install locally for testing
uv tool install ./gede-0.3.9-py3-none-any.whl

# Uninstall
uv tool uninstall gede
```

### Docker Operations

```bash
# Build Docker image
docker build --platform=linux/amd64 -t 'registry.cn-hangzhou.aliyuncs.com/slashusr/gede:16' .

# Run with Docker
docker run --rm -it -v /data/gede/data:/root/.gede --name 'gede' registry.cn-hangzhou.aliyuncs.com/slashusr/gede:16
```

## Architecture Overview

### Core Components

**Gede** is a CLI interface for multiple LLM providers built with Python. The architecture follows a modular provider pattern:

#### Main Entry Points

- `gede/gede.py` - Primary CLI interface with rich terminal UI, handles user interaction and command routing
- `gede/chatcore.py` - Core chat logic, session management, and model configuration
- `gede/commands.py` - Command parsing and execution system

#### LLM Provider System

The `gede/llm/` directory contains a unified provider system:

- `llm_provider.py` - Base provider interface and common functionality
- `providers.py` - Provider registry and factory methods
- Individual provider files (`openai_provider.py`, `anthropic_provider.py`, `qwen_provider.py`, etc.) - Implementation for specific LLM services

#### Key Abstractions

- **ChatModel** - Represents a chat session with settings, history, and provider configuration
- **LLMProvider** - Abstract base for all LLM service integrations
- **ModelSettings** - Configuration object for model parameters (temperature, reasoning effort, web search, etc.)
- **CommandContext** - Context object passed to command handlers

#### Provider Features

- Supports multiple LLM providers (OpenAI, Anthropic, Qwen, DeepSeek, Kimi, Voice Engine, Wenxin, AI302, OpenRouter)
- Unified interface for reasoning modes, web search capabilities, and model-specific settings
- Encrypted credential storage via `encrypt.py`

#### Configuration and Data

- Uses `~/.gede/` directory for user data and configuration
- Model information and capabilities stored in JSON format
- Session persistence and instruction/prompt templates

### Dependencies and Framework

- Built with Python 3.12+ using modern async/await patterns
- Uses `openai-agents` library for core agent functionality
- Rich terminal UI with `rich`, `prompt-toolkit` for interactive elements
- HTTP clients: `httpx` for async requests, `requests` for compatibility
- Data handling: `pandas`, `numpy` for data processing
- Configuration: `pydantic` for settings validation

### Key Design Patterns

- **Provider Pattern** - Pluggable LLM providers with consistent interface
- **Command Pattern** - Extensible command system for CLI operations
- **Factory Pattern** - Dynamic provider instantiation based on configuration
- **Context Object** - Shared state management across command execution

The codebase emphasizes modularity, allowing easy addition of new LLM providers while maintaining a consistent user experience across different AI services.

