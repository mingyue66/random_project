"""
Base model classes and configurations.

This module provides:
- BaseConfig: Base configuration class for all models
- BaseModel: Example/reference implementation showing how models should be structured
"""

from .model import BaseModel
from .model_config import BaseConfig

__all__ = [
    "BaseConfig",
    "BaseModel",
]
