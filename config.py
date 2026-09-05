from pathlib import Path

# Compatibility shim for tools that import configuration from the project root.
from src.config import Settings, get_settings

__all__ = ["Settings", "get_settings", "Path"]
