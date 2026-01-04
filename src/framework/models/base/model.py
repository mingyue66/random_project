"""
Example model implementation for framework models.

This module provides an example implementation showing how models should be structured
in the framework. It serves as a reference/template, not as a base class
to inherit from.

Notes:
- To use ``from_pretrained()``, concrete model classes should define ``config_class``
  pointing to their config (e.g., ``config_class = MyModelConfig``).
- This class demonstrates saving/loading MODEL CHECKPOINTS for deployment/inference.
  TRAINER CHECKPOINTS (for resuming training) are handled separately via
  utils.checkpoint.save_trainer_checkpoint/load_trainer_checkpoint.

Registering custom models/configs (for AutoConfig/AutoModel):
    The project provides dynamic registries in ``framework.auto`` so your custom
    models/configs can be discovered and loaded from directories created by
    ``save_pretrained``.

    1) Define a config class with a unique ``model_type``
        ```python
        from framework.models.base.model_config import BaseConfig

        class MyModelConfig(BaseConfig):
            model_type = "my_model"  # unique identifier used by the registry
            hidden_dim: int = 256
            num_layers: int = 12
        ```

    2) Define a model class that exposes ``config_class``
        ```python
        import torch.nn as nn

        class MyModel(nn.Module):
            config_class = MyModelConfig  # required for from_pretrained()

            def __init__(self, config: MyModelConfig):
                super().__init__()
                self.config = config
                # build layers using values from config
        ```

    3) Register the pair with the provided helpers
        ```python
        from framework.auto.auto_config import register_config, AutoConfig
        from framework.auto.auto_model import register_model, AutoModel

        # Register by model_type and module/class names (lazy import)
        register_config(
            config_type=MyModelConfig.model_type,
            module_path="my_pkg.my_model_config",  # where MyModelConfig lives
            class_name="MyModelConfig",
        )
        register_model(
            model_type=MyModelConfig.model_type,
            module_path="my_pkg.my_model",       # where MyModel lives
            class_name="MyModel",
        )
        ```

    4) Load via AutoConfig/AutoModel
        ```python
        # Given a directory saved with model.save_pretrained(exp_dir)
        # Reads exp_dir/config.json to resolve model_type and pick the right classes
        config = AutoConfig.from_pretrained(exp_dir)

        # Important: from_config(...) constructs an EMPTY model (randomly initialized)
        # using the configuration only. It does NOT load any weights.
        model = AutoModel.from_config(config)

        # To load weights, use from_pretrained(...):
        model = AutoModel.from_pretrained(exp_dir)
        ```
"""

import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from .model_config import BaseConfig


class BaseModel(nn.Module):
    """
    Example/reference implementation showing how framework models should be structured.

    This class serves as a template and reference for implementing models in the framework.
    It demonstrates best practices for:
    - Configuration management
    - Model initialization
    - Model checkpoint saving/loading (for deployment/inference)
    - Model information and utilities

    Note: This is NOT a base class to inherit from. Models should be simple nn.Module classes.
    Use this as a reference for implementing your own models.

    Important: Model checkpoints vs Trainer checkpoints
    - Model checkpoints (this class): model weights + config (for deployment/inference)
    - Trainer checkpoints: model + optimizer + scheduler + scaler + progress (for resuming training)

    Args:
        config: Model configuration object

    Example:
        ```python
        # Your models should look like this (not inheriting from BaseModel):
        class MyModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                # Initialize your model layers here

            def forward(self, inputs):
                # Implement forward pass
                return outputs
        ```
    """

    def __init__(self, config: BaseConfig):
        """
        Initialize the base model.

        Args:
            config: Model configuration object
        """
        super().__init__()
        self.config = config
        self.model_type = getattr(config, "model_type", "unknown")

        # Initialize model components
        self._init_weights()

    def _init_weights(self):
        """
        Initialize model weights.

        This method can be overridden by subclasses to implement
        custom weight initialization strategies.
        """
        # Default initialization - can be overridden
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the model.

        This is an example implementation. Your models should implement
        their own forward pass logic.

        Args:
            inputs: Input tensor
            **kwargs: Additional arguments for the forward pass

        Returns:
            Output tensor
        """
        # Example implementation - replace with your actual forward pass
        # This is just a placeholder to show the expected signature
        raise NotImplementedError(
            "This is an example class. Implement your own forward method."
        )

    def get_config(self) -> BaseConfig:
        """
        Get the model configuration.

        Returns:
            Model configuration object
        """
        return self.config

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save the model and its configuration to a directory.

        This saves a MODEL CHECKPOINT (for deployment/inference), which is different
        from TRAINER CHECKPOINTS (for resuming training). Trainer checkpoints include
        optimizer, scheduler, scaler, and training progress, while this only saves
        the model weights and configuration.

        Args:
            save_directory: Directory to save the model
            **kwargs: Additional arguments for saving

        Note:
            - This creates a MODEL CHECKPOINT for deployment/inference
            - For training resumption, use utils.checkpoint.save_trainer_checkpoint()
            - Model checkpoints contain: model weights + config
            - Trainer checkpoints contain: model + optimizer + scheduler + scaler + progress
        """
        import os
        from pathlib import Path

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save configuration
        self.config.save_pretrained(str(save_directory))

        # Save model state
        model_path = save_directory / "pytorch_model.bin"
        torch.save(self.state_dict(), model_path)

        logging.info(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load a model from a pretrained MODEL CHECKPOINT.

        This loads a MODEL CHECKPOINT (for deployment/inference), which is different
        from TRAINER CHECKPOINTS (for resuming training). This method only loads
        model weights and configuration, not training state.

        Args:
            model_path: Path to the model directory or checkpoint file
            **kwargs: Additional arguments for loading

        Returns:
            Loaded model instance

        Note:
            - This loads a MODEL CHECKPOINT for deployment/inference
            - For training resumption, use utils.checkpoint.load_trainer_checkpoint()
            - Model checkpoints contain: model weights + config
            - Trainer checkpoints contain: model + optimizer + scheduler + scaler + progress
        """
        import os
        from pathlib import Path

        model_path = Path(model_path)

        # Load configuration
        if model_path.is_dir():
            config_path = model_path / "config.json"
        else:
            config_path = model_path.parent / "config.json"

        # Concrete model classes are expected to provide ``config_class``
        # (e.g., ``config_class = MyModelConfig``). If missing, raise a clear error.
        if not hasattr(cls, "config_class"):
            raise AttributeError(
                f"{cls.__name__} is missing 'config_class'. Define it on your model class, "
                "e.g., `config_class = MyModelConfig`."
            )
        config = cls.config_class.from_pretrained(str(config_path))

        # Create model instance
        model = cls(config, **kwargs)

        # Load model weights
        weights_path = None
        if model_path.is_dir():
            # Prefer safetensors, then fallback to .bin/.pt
            for candidate in [
                model_path / "model.safetensors",
                model_path / "model.bin",
                model_path / "model.pt",
            ]:
                if candidate.exists():
                    weights_path = candidate
                    break
            if weights_path is None:
                logging.warning(f"No model weights found at {model_path}")
        else:
            weights_path = model_path

        if weights_path is not None and weights_path.exists():
            suffix = weights_path.suffix.lower()
            if suffix == ".safetensors":
                from safetensors.torch import load_file as safe_load_file

                state_dict = safe_load_file(str(weights_path), device="cpu")
            else:
                state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
            logging.info(f"Model loaded from {weights_path}")
        elif weights_path is not None:
            logging.warning(f"No model weights found at {weights_path}")

        return model

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_type": self.model_type,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "config": self.config.to_dict(),
        }

    def print_model_info(self):
        """Print model information to console."""
        info = self.get_model_info()
        print(f"Model Type: {info['model_type']}")
        print(f"Total Parameters: {info['total_parameters']:,}")
        print(f"Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"Model Config: {info['config']}")
