"""
Zipformer model configuration.

Attribution:
    This configuration corresponds to the Zipformer architecture introduced by
    the Icefall project and is adapted for the framework.
    Reference implementation (Icefall):
    https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/zipformer.py
    https://arxiv.org/pdf/2310.11230

Overview:
    This config follows a Hugging Face-style design: a plain class inheriting
    ``BaseConfig`` with explicit ``__init__`` defaults, accepting arbitrary
    ``**kwargs`` for forward compatibility, and performing light normalization of
    stage-wise lists. Single-element head-dim lists are broadcast to the number
    of encoder stacks.
"""

from typing import List, Optional

from ..base.model_config import BaseConfig


class ZipformerConfig(BaseConfig):
    """Configuration for Zipformer encoder models.

    Notes
    -----
    - The number of encoder "stacks" equals ``len(num_encoder_layers)``.
      The following per-stage lists must share this same length:
      ``downsampling_factor``, ``encoder_dim``, ``feedforward_dim``,
      ``num_heads``, ``encoder_unmasked_dim``, ``cnn_module_kernel``.
    - For ``query_head_dim``, ``value_head_dim``, and ``pos_head_dim``, a
      single-element list is allowed and will be broadcast to match the number
      of stacks.
    - ``chunk_size`` and ``left_context_frames`` control chunked causal
      training/inference and are only meaningful when ``causal=True``.

    Args
    ----
    feature_dim (int, default=80):
        Input feature dimension (e.g., FBANK bins).
    output_downsampling_factor (int, default=2):
        Additional output downsampling applied by the encoder head.
    num_encoder_layers (list[int], default=[2, 2, 3, 4, 3, 2]):
        Number of encoder layers per stack.
    downsampling_factor (list[int], default=[1, 2, 4, 8, 4, 2]):
        Extra downsampling factor per stack (on top of any frontend downsampling).
    encoder_dim (list[int], default=[192, 256, 384, 512, 384, 256]):
        Embedding dimension per stack.
    feedforward_dim (list[int], default=[576, 768, 1152, 1536, 1152, 768]):
        Hidden dimension of feedforward modules per stack.
    warmup_batches (float, default=4000.0):
        Number of batches to warm up (affects dropout scheduling in some setups).
    dropout (float | None, default=None):
        Global dropout rate; if ``None``, uses module defaults.
    num_heads (list[int], default=[4, 4, 4, 8, 4, 4]):
        Attention head count per stack.
    query_head_dim (list[int], default=[32]):
        Query/key dimension per head; a single value will be broadcast per stack.
    value_head_dim (list[int], default=[12]):
        Value dimension per head; a single value will be broadcast per stack.
    pos_head_dim (list[int], default=[4]):
        Positional-encoding projection dimension per head; broadcast if length 1.
    pos_dim (int, default=48):
        Pre-projection positional-encoding vector dimension.
    encoder_unmasked_dim (list[int], default=[192, 192, 256, 256, 256, 192]):
        Unmasked dimension used for per-frame dropout per stack.
    cnn_module_kernel (list[int], default=[31, 31, 15, 15, 15, 31]):
        Convolution kernel size per stack.
    causal (bool, default=False):
        Enable chunkwise causal convolution for streaming-like behavior.
    chunk_size (list[int], default=[16, 32, 64, -1]):
        Candidate chunk sizes; ``-1`` disables chunking. Only used if ``causal``.
    left_context_frames (list[int], default=[64, 128, 256, -1]):
        Left context (in frames) for causal training; rounded to chunk multiples.
    **kwargs:
        Extra parameters preserved for forward compatibility.
    """

    model_type: str = "zipformer"

    def __init__(
        self,
        feature_dim: int = 80,
        output_downsampling_factor: int = 2,
        num_encoder_layers: List[int] = [2, 2, 3, 4, 3, 2],
        downsampling_factor: List[int] = [1, 2, 4, 8, 4, 2],
        encoder_dim: List[int] = [192, 256, 384, 512, 384, 256],
        feedforward_dim: List[int] = [576, 768, 1152, 1536, 1152, 768],
        warmup_batches: float = 4000.0,
        dropout: Optional[float] = None,
        num_heads: List[int] = [4, 4, 4, 8, 4, 4],
        query_head_dim: List[int] = [32],
        value_head_dim: List[int] = [12],
        pos_head_dim: List[int] = [4],
        pos_dim: int = 48,
        encoder_unmasked_dim: List[int] = [192, 192, 256, 256, 256, 192],
        cnn_module_kernel: List[int] = [31, 31, 15, 15, 15, 31],
        causal: bool = False,
        chunk_size: List[int] = [16, 32, 64, -1],
        left_context_frames: List[int] = [64, 128, 256, -1],
        **kwargs,
    ):
        """HF-style initializer with explicit defaults and forward-compatible kwargs."""
        # Preserve unknown keys for forward compatibility
        super().__init__(**kwargs)

        # Assign known fields
        self.feature_dim = feature_dim
        self.output_downsampling_factor = output_downsampling_factor

        self.num_encoder_layers = num_encoder_layers
        self.downsampling_factor = downsampling_factor
        self.encoder_dim = encoder_dim
        self.feedforward_dim = feedforward_dim

        self.warmup_batches = warmup_batches
        self.dropout = dropout

        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.value_head_dim = value_head_dim
        self.pos_head_dim = pos_head_dim
        self.pos_dim = pos_dim

        self.encoder_unmasked_dim = encoder_unmasked_dim
        self.cnn_module_kernel = cnn_module_kernel
        self.causal = causal
        self.chunk_size = chunk_size
        self.left_context_frames = left_context_frames


# Preset configs
zipformer_base_config = ZipformerConfig()
zipformer_large_config = ZipformerConfig(
    encoder_dim=[192, 256, 512, 768, 512, 256],
    feedforward_dim=[576, 768, 1536, 2304, 1536, 768],
    num_encoder_layers=[2, 2, 4, 5, 4, 2],
)
