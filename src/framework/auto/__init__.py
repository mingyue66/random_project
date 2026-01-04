"""
Framework Auto Classes - HuggingFace-style model loading interface.

Provides automatic model and configuration loading based on
configuration files, similar to HuggingFace Transformers.

Note: Project-specific tokenizer factories have been deprecated in favor
of using ``transformers.AutoTokenizer`` directly.
"""

from .auto_config import AutoConfig, list_available_configs, register_config
from .auto_model import AutoModel, list_available_models, register_model

__all__ = [
    "AutoConfig",
    "AutoModel",
    "register_config",
    "register_model",
    "list_available_configs",
    "list_available_models",
]
