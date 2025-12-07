# Contributing to Gede

Thank you for your interest in contributing to Gede! We welcome contributions from the community and appreciate your help in making this project better.

## Getting Started

### Prerequisites

- Python 3.12 or higher
- `uv` package manager
- Git

### Development Setup

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/gede.git
   cd gede
   ```

3. **Create a development branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

4. **Install dependencies**:
   ```bash
   uv sync
   ```

5. **Install optional dependencies (optional)**:
   
   For advanced tracing with Arize Phoenix:
   ```bash
   uv pip install -e ".[arize-trace]"
   ```

6. **Run the application**:
   ```bash
   python3 -m gede.gede
   ```

## Development Workflow

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular

### Making Changes

1. **Make your changes** in your feature branch
2. **Test your changes** thoroughly
3. **Commit with clear messages**:
   ```bash
   git commit -m "feat: add new feature description"
   git commit -m "fix: resolve issue description"
   ```

### Commit Message Format

We follow conventional commits. Use the following prefixes:

- `feat:` - A new feature
- `fix:` - A bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, missing semicolons, etc.)
- `refactor:` - Code refactoring without feature changes
- `perf:` - Performance improvements
- `test:` - Adding or updating tests
- `chore:` - Dependency updates, build changes

### Testing

Before submitting a PR:

1. **Test your changes manually**:
   ```bash
   python3 -m gede.gede --help
   python3 -m gede.gede  # Test the application
   ```

2. **Test tracing functionality** (if modified):
   
   Test with trace enabled:
   ```bash
   # Test with Arize Phoenix (if installed)
   python3 -m gede.gede --trace --log-level DEBUG
   
   # Test fallback to OpenAI tracing (set OPENAI_API_KEY first)
   uv run pip uninstall arize-phoenix-otel -y
   OPENAI_API_KEY=your_key python3 -m gede.gede --trace --log-level DEBUG
   
   # Reinstall for development
   uv pip install "arize-phoenix-otel>=0.13.1"
   ```

3. **Check for syntax errors**:
   ```bash
   python3 -m py_compile gede/**/*.py
   ```

## Submitting Changes

### Pull Request Process

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** on GitHub with:
   - Clear title describing the change
   - Detailed description of what changed and why
   - Reference to any related issues (e.g., "Closes #123")
   - Screenshots for UI changes (if applicable)

3. **PR Description Template**:
   ```markdown
   ## Description
   Brief explanation of the changes

   ## Related Issue
   Closes #(issue number)

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update

   ## Testing
   How to test these changes

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-reviewed code
   - [ ] Tested manually
   ```

## Reporting Issues

### Bug Reports

Include the following information:

1. **Environment**:
   ```
   - OS: (e.g., macOS 14.1, Ubuntu 22.04)
   - Python version: (e.g., 3.12.1)
   - Gede version: (output of `gede --version`)
   ```

2. **Steps to reproduce**
3. **Expected behavior**
4. **Actual behavior**
5. **Error logs** (if applicable)
6. **Screenshots** (if applicable)

### Feature Requests

Include:

1. **Use case** - Why do you need this feature?
2. **Proposed solution** - How should it work?
3. **Alternative solutions** - Other approaches considered?

## Code Review Process

Your PR will be reviewed by maintainers. We may:

- Request changes
- Ask clarifying questions
- Suggest improvements

Please be responsive to feedback and make requested changes. Once approved, your changes will be merged!

## Areas for Contribution

We appreciate contributions in these areas:

- **Features**: New LLM providers, tools, or functionality
- **Bug fixes**: Issues with existing features
- **Documentation**: README, docstrings, examples
- **Tests**: Test coverage improvements
- **Performance**: Optimization suggestions
- **Translations**: Internationalization support

## Optional Dependencies

### Adding New Optional Dependencies

When adding new optional features, follow this pattern:

1. Add the dependency to `[project.optional-dependencies]` in `pyproject.toml`:
   ```toml
   [project.optional-dependencies]
   my-feature = ["dependency>=1.0.0"]
   ```

2. In your code, handle the import gracefully:
   ```python
   try:
       from some_package import feature
       HAS_FEATURE = True
   except ImportError:
       HAS_FEATURE = False
   ```

3. Provide fallback behavior or clear error messages when the optional dependency is not installed.

4. Update documentation in README.md and CONTRIBUTING.md with installation instructions.

### Current Optional Dependencies

- **arize-trace**: Enables Arize Phoenix tracing capabilities when `--trace` flag is used
  - Install with: `uv pip install "gede[arize-trace]"`
  - Usage: `python3 -m gede.gede --trace`

## Questions or Need Help?

- Open an issue with the `question` label
- Check existing issues and discussions
- Review the README and documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to Gede! 🎉
