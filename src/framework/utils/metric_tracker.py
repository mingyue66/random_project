"""
Metrics tracking utilities for Framework training and evaluation.

This module provides the MetricsTracker class for collecting, aggregating, and logging
training metrics with support for different normalization strategies and distributed training.

Key Features:
- Multiple normalization types: batch_avg, frame_avg, sample_avg, sum
- Distributed training support with proper all-reduce operations
- TensorBoard integration for logging
- Flexible input handling (dictionaries or MetricsTracker objects)
- Exponential moving average support for metric smoothing

Usage Examples:
    ```python
    # Basic usage with dictionary
    metrics_dict = {"loss": 0.5, "accuracy": 0.95}
    tracker = MetricsTracker.from_dict(metrics_dict, batch_size=32)

    # Update with another dictionary
    tracker.update({"loss": 0.4, "accuracy": 0.96}, batch_size=16)

    # TensorBoard logging
    tracker.write_summary(tb_writer, "train/", step)

    # Distributed training
    if world_size > 1:
        tracker.reduce(device)
    ```

Normalization Types:
- batch_avg: Average across batches (most common for loss, accuracy)
- frame_avg: Average across frames (useful for audio tasks)
- sample_avg: Average across samples (useful for per-sample metrics)
- sum: Sum values (useful for counts, total loss)

The MetricsTracker automatically handles proper weighting during distributed training
and provides consistent metric aggregation across different normalization strategies.
"""

import logging
import math
from typing import Dict, Literal, Union

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

NormType = Literal["batch_avg", "frame_avg", "sample_avg", "sum"]
Number = Union[int, float]


class MetricsTracker:
    """
    A flexible metrics tracker for training and evaluation with distributed training support.

    This class provides comprehensive metric tracking with support for different normalization
    strategies, distributed training, and TensorBoard logging. It can handle both individual
    metric values and batch-level aggregations.

    Key Features:
    - Multiple normalization types (batch_avg, frame_avg, sample_avg, sum)
    - Automatic batch/frame/sample count tracking
    - Distributed training support with proper all-reduce operations
    - TensorBoard integration
    - Flexible input handling (dictionaries or MetricsTracker objects)
    - Exponential moving average support

    Attributes:
        _values: Dictionary storing metric values
        _norms: Dictionary storing normalization types for each metric

    Example:
        ```python
        # Create from dictionary
        tracker = MetricsTracker.from_dict({"loss": 0.5, "accuracy": 0.95}, batch_size=32)

        # Update with new metrics
        tracker.update({"loss": 0.4, "accuracy": 0.96}, batch_size=16)

        # Log to TensorBoard
        tracker.write_summary(tb_writer, "train/", step)

        # Distributed training
        if world_size > 1:
            tracker.reduce(device)
        ```
    """

    def __init__(self):
        """Initialize an empty MetricsTracker."""
        self._values: Dict[str, float] = {}
        self._norms: Dict[str, NormType] = {}

    def set_value(self, key: str, value: Number, normalization: NormType = "batch_avg"):
        """Set a pre-normalized metric value."""
        self._values[key] = float(value)
        self._norms[key] = normalization

    @classmethod
    def from_dict(
        cls,
        metrics_dict: Dict[str, float],
        batch_size: int = 1,
        frame_count: int = 0,
        sample_count: int = 0,
        default_normalization: NormType = "batch_avg",
    ) -> "MetricsTracker":
        """
        Create MetricsTracker from dictionary with automatic count tracking.

        Args:
            metrics_dict: Dictionary of metric names to values
            batch_size: Number of batches (default: 1)
            frame_count: Number of frames (default: 0, not tracked)
            sample_count: Number of samples (default: 0, not tracked)
            default_normalization: Default normalization type for metrics

        Returns:
            MetricsTracker instance with metrics and counts set

        Example:
            ```python
            metrics_dict = {"loss": 0.5, "accuracy": 0.95}
            tracker = MetricsTracker.from_dict(metrics_dict, batch_size=32)
            ```
        """
        tracker = cls()

        # Add metric values with default normalization
        for key, value in metrics_dict.items():
            tracker.set_value(key, value, default_normalization)

        # Add batch count (always tracked)
        tracker.set_value("batches", batch_size, "sum")

        # Add frame count if provided
        if frame_count > 0:
            tracker.set_value("frames", frame_count, "sum")

        # Add sample count if provided
        if sample_count > 0:
            tracker.set_value("samples", sample_count, "sum")

        return tracker

    def update(
        self,
        other: Union["MetricsTracker", Dict[str, float]],
        reset_interval: int = -1,
        batch_size: int = 1,
    ):
        """
        Update this MetricsTracker with values from another MetricsTracker or dictionary.

        Args:
            other: MetricsTracker instance or dictionary of metric values
            reset_interval: Exponential moving average decay factor (>0) or no decay (-1)
            batch_size: Batch size for dictionary inputs (ignored for MetricsTracker inputs)
        """
        # Convert dictionary to MetricsTracker if needed
        if isinstance(other, dict):
            other = self.from_dict(other, batch_size=batch_size)

        if reset_interval > 0:
            alpha = 1 - 1.0 / reset_interval
        else:
            alpha = 1

        # Step 1: update actual metrics
        for k, v in other._values.items():
            if k in ("batches", "frames", "samples"):
                continue

            if math.isnan(v) or math.isinf(v):
                logging.warning(
                    f"[MetricsTracker] Invalid value in update('{k}'): {v}. Skipping."
                )
                continue

            norm = other._norms[k]
            self._norms[k] = norm

            if norm == "sum":
                self._values[k] = self._values.get(k, 0.0) + v

            elif norm == "batch_avg":
                prev_batches = self._values.get("batches", 0.0)
                new_batches = other._values.get("batches", 1.0)

                if k in self._values:
                    prev = self._values[k]
                    self._values[k] = (
                        alpha * prev * prev_batches + v * new_batches
                    ) / (alpha * prev_batches + new_batches)
                else:
                    self._values[k] = v

            elif norm == "frame_avg":
                prev_frames = self._values.get("frames", 0.0)
                new_frames = other._values.get("frames", 0.0)

                if new_frames > 0:
                    if k in self._values:
                        prev = self._values[k]
                        self._values[k] = (
                            alpha * prev * prev_frames + v * new_frames
                        ) / (alpha * prev_frames + new_frames)
                    else:
                        self._values[k] = v

            elif norm == "sample_avg":
                prev_samples = self._values.get("samples", 0.0)
                new_samples = other._values.get("samples", 0.0)

                if new_samples > 0:
                    if k in self._values:
                        prev = self._values[k]
                        self._values[k] = (
                            alpha * prev * prev_samples + v * new_samples
                        ) / (alpha * prev_samples + new_samples)
                    else:
                        self._values[k] = v

            else:
                raise ValueError(f"Unsupported normalization: {norm}")

        # Step 2: update global counts in the end
        # Always increment batches (default = 1)
        self._values["batches"] = alpha * self._values.get(
            "batches", 0.0
        ) + other._values.get("batches", 1.0)
        self._norms["batches"] = "sum"
        for meta_key in ("frames", "samples"):
            if meta_key in other._values:
                self._values[meta_key] = (
                    alpha * self._values.get(meta_key, 0.0) + other._values[meta_key]
                )
                self._norms[meta_key] = "sum"

    def write_summary(self, tb_writer: SummaryWriter, prefix: str, step: int):
        for k, v in self._values.items():
            tb_writer.add_scalar(f"{prefix}{k}", v, step)

    def reduce(self, device):
        """All-reduce values across DDP workers, using correct weighting for avg metrics."""

        if not dist.is_available() or not dist.is_initialized():
            return

        keys = sorted(self._values.keys())
        values_to_reduce = []

        for k in keys:
            v = self._values[k]
            norm = self._norms.get(k, "sum")
            if norm == "sample_avg":
                v *= self._values.get("samples", 1.0)
            elif norm == "frame_avg":
                v *= self._values.get("frames", 1.0)
            elif norm == "batch_avg":
                v *= self._values.get("batches", 1.0)
            # "sum" → use as-is
            values_to_reduce.append(v)

        tensor = torch.tensor(values_to_reduce, dtype=torch.float32, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        # Store back and re-normalize
        for i, k in enumerate(keys):
            norm = self._norms.get(k, "sum")
            self._values[k] = tensor[i].item()

        # Now normalize
        for k, norm in self._norms.items():
            if k in ("samples", "frames", "batches"):
                continue
            if norm == "sample_avg":
                self._values[k] /= self._values.get("samples", 1.0)
            elif norm == "frame_avg":
                self._values[k] /= self._values.get("frames", 1.0)
            elif norm == "batch_avg":
                self._values[k] /= self._values.get("batches", 1.0)

    def __str__(self):
        return ", ".join(
            f"{self._norms[k]}_{k}={v:.4g}" for k, v in self._values.items()
        )
