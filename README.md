# Gede

> 🚀 A powerful and feature-rich CLI for interacting with multiple LLM providers

Gede is a powerful command-line interface that seamlessly integrates with multiple
LLM providers including OpenAI, Anthropic, and DeepSeek. It features local chat
history management, built-in tool calling capabilities, and MCP (Model Context
Protocol) integration for enhanced AI interactions.

## Features

- 🤖 **Multi-Provider Support**: OpenAI, Anthropic, DeepSeek, Qwen, Baidu, OpenRouter, Moonshot, and more
- 💬 **Chat Management**: Create public, private (encrypted), and cloned conversations
- 🛠️ **Rich Tools Ecosystem**: Built-in web search, URL reading, and custom tools
- 🔌 **MCP Server Integration**: Connect to Model Context Protocol servers
- 📦 **Profile Support**: Manage multiple configurations with profiles
- 🌐 **Web Search**: Enable AI model's built-in web search capability

## Quick Start

### Prerequisites

- Python 3.12 or higher
- `uv` package manager

## Install

TODO

### Quick Example

```bash
# Start a new chat
gede

# Or start with a specific model
gede --model openai:gpt-4o

# Start in private mode
gede --private

# Use with tools enabled
gede --tools web_search,now
```

## Slash Commands

When using Gede, you can use slash commands to perform various operations. Type `/help` to see all commands, or `/help KEYWORD` to search for specific commands.

### Chat Management

| Command        | Description                                                                        |
| -------------- | ---------------------------------------------------------------------------------- |
| `/new`         | Start a new public chat (plain text)                                               |
| `/new-private` | Start a new private chat (password-encrypted)                                      |
| `/chat-info`   | Display current chat details (ID, title, model, message count, tools, MCP servers) |
| `/clone-chat`  | Create a new chat with same settings (instruction, model, parameters)              |
| `/quit`        | Exit the application (unsaved private chats won't persist)                         |

### Instruction & Prompt Management

| Command                   | Description                                                                |
| ------------------------- | -------------------------------------------------------------------------- |
| `/set-instruction <TEXT>` | Set system instruction. Use `\\` for multi-line mode (Esc+Enter to submit) |
| `/get-instruction`        | Display current system instruction                                         |
| `/select-instruction`     | Choose from predefined instructions in `~/.gede/instructions/`           |
| `/select-prompt`          | Select a predefined prompt as input message from `~/.gede/prompts/`      |

### Model Settings

| Command                                 | Description                                                                                                                              |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `/select-llm [PROVIDER] [--no-cache]`   | Switch AI model. Use `--no-cache` to refresh model list                                                                                  |
| `/set-message-num NUMBER`               | Control chat history length (0 = all messages)                                                                                           |
| `/set-model-settings KEY VALUE`         | Adjust parameters: temperature (0-2), top_p (0-1), max_tokens, frequency_penalty (-2 to 2), presence_penalty (-2 to 2), reasoning_effort |
| `/get-model-settings`                   | Display current model parameters                                                                                                         |
| `/set-model-reasoning <LEVEL>`          | Control reasoning depth: minimal, low, medium, high, auto, or off                                                                        |
| `/set-model-web-search <on\|off\|auto>` | Toggle web search capability                                                                                                             |

### File Operations

| Command              | Description                                                                                          |
| -------------------- | ---------------------------------------------------------------------------------------------------- |
| `/save`              | Save current chat. Public: auto-saved with generated title. Private: requires password               |
| `/load-chat`         | Load a public chat from `~/.gede/chats/public/` (interactive selection)                            |
| `/load-private-chat` | Load private chat from `~/.gede/chats/private/` (password required)                                |
| `/export <FILEPATH>` | Export chat to text file. Relative paths save to `~/.gede/chats/exports/` or specificed file path. |

### Tools & MCP

| Command         | Description                                                       |
| --------------- | ----------------------------------------------------------------- |
| `/select-tools` | Enable/disable built-in tools (Space to toggle, Enter to confirm) |
| `/select-mcp`   | Connect to MCP servers (Space to toggle, Enter to confirm)        |

### Utility

| Command           | Description                            |
| ----------------- | -------------------------------------- |
| `/cleanup`        | Clear terminal screen                  |
| `/help [KEYWORD]` | Show all commands or search by keyword |

## CLI Usage

### Command Line Arguments

Gede supports the following command line arguments:

- `--profile <profile_name>`: Use specified configuration profile (default: default)
- `--log-level <level>`: Set log level, options: DEBUG, INFO, WARNING, ERROR, CRITICAL
- `--model <provider_id:model_id>`: Specify default model, e.g.: `openai:gpt-4o`
- `--instruction <text>`: Set system prompt
- `--private`: Start private session
- `--reasoning-effort <effort>`: Set reasoning mode, options: minimal, low, medium, high, off, auto
- `--web-search <mode>`: Enable or disable model's built-in web search, options: on, off, auto
- `--tools <tool_list>`: Set enabled tools list, multiple tools separated by commas, e.g.: `web_search,now,read_page`
- `--trace`: Enable trace mode for analyzing detailed execution information of agent calls. Uses Arize Phoenix if the `arize-trace` extra is installed, otherwise uses OpenAI's default tracing (requires `OPENAI_API_KEY`)
- `--mcp <server_list>`: Enable MCP servers, multiple servers separated by commas

### Usage Examples

```bash
# Start with default configuration
gede

# Start with specified model
gede --model openai:gpt-4o

# Enable tools and private mode
gede --tools web_search,now --private

# Set reasoning mode and log level
gede --reasoning-effort high --log-level DEBUG

# Use specific profile
gede --profile my_profile
```

## Configuration

### Storage

On first launch, Gede will automatically create a configuration directory at `~/.gede/` with:

- `config`
  - `.env` - Configuration file for API keys
  - `mcp.json` - MCP server confirugation
  - `profiles.json` - Profile confirugation
- `chats/public/` - Public chat storage
- `chats/private/` - Encrypted private chat storage
- `instructions/` - Custom system instructions
- `prompts/` - Predefined prompts

Gede uses environment variables to store API keys for various LLM providers. The configuration file is located at `~/.gede/config/.env`. Edit this file to add your actual API keys.

### Supported Providers

When you first run Gede, a default config file will be automatically created. Supported providers include:

- **302.ai**: `AI302_API_KEY`
- **OpenRouter**: `OPENROUTER_API_KEY`
- **OpenAI**: `OPENAI_API_KEY`
- **Anthropic**: `ANTHROPIC_API_KEY`
- **Baidu (ERNIE)**: `WENXIN_API_KEY`
- **SiliconFlow**: `SILICONFLOW_API_KEY`
- **Aliyun (Qwen)**: `QWEN_API_KEY`
- **VoiceEngine (Doubao)**: `DOUBAO_API_KEY`
- **DeepSeek**: `DEEPSEEK_API_KEY`
- **Moonshot (Kimi)**: `MOONSHOT_API_KEY`

The config file also supports:

- **Generate Title Model**: Use specific model for chat title generation
- **Phoenix Tracing**: Configure observability with Arize Phoenix

### Profile

Gede supports profile management to save and reuse your preferred configurations. The profile configuration file is located at `~/.gede/config/profiles.json`.

#### Profile Structure

Each profile can contain the following settings:

- `model`: Default model to use (format: `provider:model_id`)
- `instruction`: System instruction/prompt
- `private`: Whether to start in private mode (boolean)
- `reasoning_effort`: Reasoning depth level (`minimal`, `low`, `medium`, `high`, `auto`, `off`)
- `web_search`: Web search mode (`on`, `off`, `auto`)
- `tools`: List of enabled tools (e.g., `["web_search", "now", "read_page"]`)
- `trace`: Enable trace mode (boolean)
- `log_level`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`)
- `mcp`: List of MCP servers to auto-connect

#### Example Configuration

```json
{
  "default": {
    "model": "openai:gpt-4o",
    "instruction": "You are a helpful assistant.",
    "private": false,
    "reasoning_effort": "medium",
    "web_search": "auto",
    "tools": ["web_search", "now", "read_page"],
    "trace": false,
    "log_level": "INFO"
  },
  "coding": {
    "model": "anthropic:claude-sonnet-4-20250514",
    "instruction": "You are an expert programming assistant.",
    "reasoning_effort": "high",
    "tools": ["web_search", "read_page"],
    "log_level": "DEBUG"
  },
  "research": {
    "model": "openai:gpt-4o",
    "instruction": "You are a research assistant specialized in finding and analyzing information.",
    "web_search": "on",
    "tools": ["web_search", "read_page"],
    "mcp": ["filesystem"]
  }
}
```

#### Usage

```bash
# Use default profile
gede

# Use specific profile
gede --profile coding

# Use profile and override settings
gede --profile research --model deepseek:deepseek-reasoner
```

**Note**: Command-line arguments will override profile settings for the current session.

### MCP

The MCP configuration file is located at `~/.gede/config/mcp.json`. It allows you to define multiple MCP servers that Gede can connect to.

#### STDIO Server

Connects to a local process via standard input/output.

- `command` (required): The executable command to run.
- `args` (optional): List of arguments for the command.
- `env` (optional): Dictionary of environment variables.
- `cwd` (optional): Working directory for the process.
- `auto_select` (optional, default: `false`): Whether to automatically select this server on startup.
- `enable` (optional, default: `true`): Whether this server is enabled.

#### Remote Server (SSE / Streamable HTTP)

Connects to a remote MCP server.

- `type` (required): Must be either `sse` or `streamable-http`.
- `url` (required): The URL of the server endpoint.
- `headers` (optional): Dictionary of HTTP headers.
- `note` (optional): Description or note for the server.
- `auto_select` (optional, default: `false`): Whether to automatically select this server on startup.
- `enable` (optional, default: `true`): Whether this server is enabled.

#### Example Configuration

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/username/Desktop"
      ],
      "auto_select": true
    },
    "remote-echo": {
      "type": "sse",
      "url": "https://example.com/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN"
      },
      "note": "My remote MCP server",
      "auto_select": false,
      "enable": true
    }
  }
}
```

### Build-in Tools

- web_search: exa web search
- read_page: URL reading
- now: time information

### Optional Dependencies

Gede supports optional extensions for enhanced functionality:

#### Arize Phoenix Tracing (`arize-trace`)

Enable advanced tracing and observability with [Arize Phoenix](https://phoenix.arize.com/). This extension is used when you enable trace mode with the `--trace` flag.

**Installation:**

```bash
uv pip install "gede[arize-trace]"
```

**Usage:**

When the `arize-trace` extension is installed and `--trace` is enabled, Gede will automatically use Arize Phoenix for tracing:

```bash
gede --trace
```

If the extension is not installed, Gede will fall back to OpenAI's built-in tracing (if `OPENAI_API_KEY` is set).

**Configuration:**

To use Arize Phoenix, edit `~/.gede/config/.env` and configure:

```env
# Phoenix trace endpoint (customize with your project token if needed)
PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/s/your-project-token/v1/traces
```

If not configured, it defaults to `https://app.phoenix.arize.com`.

## Develop

```bash
# Clone the repository
git clone https://github.com/adow/gede.git
cd gede

# Install dependencies using uv
uv sync

# Run Gede
python3 -m gede.gede
```

### Project Structure

```
gede/
├── gede/
│   ├── commands/              # Slash command implementations
│   │   ├── base.py           # Command base class
│   │   ├── chat_commands.py  # Chat management commands
│   │   ├── model_commands.py # Model selection and settings
│   │   ├── file_commands.py  # File operations (save, load, export)
│   │   └── ...              # Other command modules
│   ├── llm/
│   │   ├── providers.py       # LLM provider registry
│   │   ├── *_provider.py     # Individual provider implementations
│   │   │   ├── openai_provider.py
│   │   │   ├── anthropic_provider.py
│   │   │   ├── deepseek_provider.py
│   │   │   └── ...          # Other providers
│   │   ├── tools/            # Built-in tools
│   │   │   ├── web_search.py
│   │   │   ├── read_url_tool.py
│   │   │   └── time_tool.py
│   │   └── mcp/              # Model Context Protocol integration
│   ├── chatcore.py           # Core chat logic
│   ├── gede.py             # Main CLI entry point
│   ├── config.py             # Configuration management
│   ├── encrypt.py            # Encryption utilities
│   ├── profiles.py           # Profile management
│   └── top.py                # Top-level utilities
├── CONTRIBUTING.md           # Contribution guidelines
├── CODE_OF_CONDUCT.md       # Community code of conduct
├── CHANGELOG.md             # Version history
├── LICENSE                  # MIT License
├── pyproject.toml           # Python project configuration
├── Dockerfile               # Docker configuration
└── README.md               # This file
```

## Technology Stack

- **Language**: Python 3.12+
- **CLI Framework**: rich, inquirer, prompt-toolkit,
- **Encryption**: cryptography
- **HTTP Client**: httpx
- **Agent Framework**: OpenAI Agent
- **Build**: uv

## Security

- Password-protected private chats with AES encryption
- User data stays local by default - chat history is ephemeral and only persisted when explicitly saved using `/save` command

## Community

- 📝 [Issues & Discussions](https://github.com/adow/gede/issues)
- 🤝 [Contributing Guidelines](CONTRIBUTING.md)
- 📖 [Code of Conduct](CODE_OF_CONDUCT.md)
- 📊 [Changelog](CHANGELOG.md)

## License

## Acknowledgments

Thanks to all contributors and the open-source community for support and feedback!

## Disclaimer

Gede is provided "as-is" for research and personal use. Users are responsible for complying with LLM provider terms of service and applicable laws when using this tool.
