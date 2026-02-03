# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**始终使用中文来回复用户和完成计划**

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
