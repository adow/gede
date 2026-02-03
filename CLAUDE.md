# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**始终使用中文来回复用户和完成计划**

## 正在重构

- 正在开发入口文件 gede2.py (代替 gede.py)
- 使用 my_llmkit 提供 llm chat, 以及模型能力信息
- llm/providers2 中的内容代替 llm 下各个 provider 的原有功能
- chatcore2.py 代替 chatcore.py
- context.py 代替 commands 中的 CommandContext

## Prerequisites

- Python 3.10 or higher
- `uv` package manager

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
