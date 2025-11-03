"""Configuration utilities for loading API keys and settings.

This module provides utilities for loading configuration from .env files
and environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def find_project_root() -> Path:
    """Find the project root directory (contains .env file or pyproject.toml).

    Returns:
        Path to project root directory.

    Raises:
        RuntimeError: If project root cannot be found.
    """
    current = Path.cwd()

    # Try to find .env or pyproject.toml up to 5 levels up
    for _ in range(5):
        if (current / ".env").exists() or (current / "pyproject.toml").exists():
            return current
        if current.parent == current:
            break
        current = current.parent

    # Default to current working directory
    return Path.cwd()


def load_env_file(env_path: str | Path | None = None) -> dict[str, str]:
    """Load environment variables from .env file.

    Args:
        env_path: Path to .env file. If None, searches for .env in project root.

    Returns:
        Dictionary of environment variables loaded from file.
    """
    if env_path is None:
        env_path = find_project_root() / ".env"
    else:
        env_path = Path(env_path)

    if not env_path.exists():
        return {}

    env_vars = {}

    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Parse KEY=VALUE
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                env_vars[key] = value

    return env_vars


def load_dotenv(env_path: str | Path | None = None, override: bool = False) -> None:
    """Load .env file into os.environ.

    Args:
        env_path: Path to .env file. If None, searches for .env in project root.
        override: If True, override existing environment variables.
    """
    env_vars = load_env_file(env_path)

    for key, value in env_vars.items():
        if override or key not in os.environ:
            os.environ[key] = value


def get_api_key(
    key_name: str, env_var: str | None = None, required: bool = False
) -> str | None:
    """Get API key from environment or .env file.

    Args:
        key_name: Name of the API key (e.g., "OpenAI", "FRED").
        env_var: Environment variable name. If None, derived from key_name.
        required: If True, raise error if key not found.

    Returns:
        API key string, or None if not found and not required.

    Raises:
        ValueError: If required=True and key not found.

    Example:
        >>> get_api_key("OpenAI")  # Looks for OPENAI_API_KEY
        'sk-...'
        >>> get_api_key("Custom", env_var="MY_CUSTOM_KEY")
        'abc123'
    """
    # Ensure .env is loaded
    load_dotenv()

    # Derive environment variable name if not provided
    if env_var is None:
        env_var = f"{key_name.upper().replace(' ', '_')}_API_KEY"

    # Try to get from environment
    api_key = os.getenv(env_var)

    if api_key is None or api_key == "":
        if required:
            raise ValueError(
                f"{key_name} API key not found. "
                f"Please set {env_var} in your .env file or environment."
            )
        return None

    return api_key


class Config:
    """Configuration manager for AlphaLab.

    Automatically loads .env file and provides easy access to configuration.

    Attributes:
        openai_api_key: OpenAI API key
        anthropic_api_key: Anthropic API key
        fred_api_key: FRED API key
        fmp_api_key: Financial Modeling Prep API key
        default_llm_provider: Default LLM provider (openai or anthropic)
        default_llm_model: Default LLM model (optional)
        data_cache_dir: Directory for caching data
        debug: Enable debug logging

    Example:
        >>> config = Config()
        >>> config.openai_api_key
        'sk-...'
        >>> config.default_llm_provider
        'openai'
    """

    def __init__(self, env_path: str | Path | None = None):
        """Initialize configuration.

        Args:
            env_path: Path to .env file. If None, searches for .env in project root.
        """
        # Load .env file
        load_dotenv(env_path)

        # LLM API keys
        self.openai_api_key = get_api_key("OpenAI")
        self.anthropic_api_key = get_api_key("Anthropic")

        # Data source API keys
        self.fred_api_key = get_api_key("FRED")
        self.fmp_api_key = get_api_key("FMP")

        # Optional data sources
        self.alpha_vantage_api_key = get_api_key("Alpha Vantage", "ALPHA_VANTAGE_API_KEY")
        self.polygon_api_key = get_api_key("Polygon", "POLYGON_API_KEY")
        self.quandl_api_key = get_api_key("Quandl", "QUANDL_API_KEY")

        # Configuration options
        self.default_llm_provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
        self.default_llm_model = os.getenv("DEFAULT_LLM_MODEL") or None
        self.data_cache_dir = os.getenv("DATA_CACHE_DIR", "data/cache")
        self.debug = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key.
            default: Default value if key not found.

        Returns:
            Configuration value.
        """
        return getattr(self, key, default)

    def has_llm_key(self, provider: str | None = None) -> bool:
        """Check if LLM API key is available.

        Args:
            provider: LLM provider (openai or anthropic). If None, uses default.

        Returns:
            True if API key is available.
        """
        if provider is None:
            provider = self.default_llm_provider

        if provider == "openai":
            return self.openai_api_key is not None
        elif provider == "anthropic":
            return self.anthropic_api_key is not None
        else:
            return False

    def has_data_key(self, source: str) -> bool:
        """Check if data source API key is available.

        Args:
            source: Data source name (fred, fmp, alpha_vantage, polygon, quandl).

        Returns:
            True if API key is available.
        """
        key_attr = f"{source.lower()}_api_key"
        return getattr(self, key_attr, None) is not None

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Config("
            f"llm={self.default_llm_provider}, "
            f"openai={'[OK]' if self.openai_api_key else '[FAIL]'}, "
            f"anthropic={'[OK]' if self.anthropic_api_key else '[FAIL]'}, "
            f"fred={'[OK]' if self.fred_api_key else '[FAIL]'}, "
            f"fmp={'[OK]' if self.fmp_api_key else '[FAIL]'}"
            f")"
        )


# Global config instance (lazy loaded)
_config: Config | None = None


def get_config() -> Config:
    """Get global configuration instance.

    Returns:
        Global Config instance.

    Example:
        >>> from alphalab.utils.config import get_config
        >>> config = get_config()
        >>> config.openai_api_key
        'sk-...'
    """
    global _config
    if _config is None:
        _config = Config()
    return _config
