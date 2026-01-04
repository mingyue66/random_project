"""
Framework Training Framework

Provides base trainer class and utilities for training audio models.
Task-specific trainers should inherit from BaseTrainer.
"""

from .ddp_trainer import BaseTrainer

__all__ = ["BaseTrainer"]
