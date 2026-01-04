"""
ASR model configuration.

This config describes a simple transducer/CTC ASR stack with a pluggable
encoder configuration (e.g., Zipformer). It follows the HF-style explicit
__init__ defaults and stores arbitrary extra kwargs via BaseConfig.
"""

from ..base.model_config import BaseConfig
from ..zipformer.model_config import ZipformerConfig


class AsrConfig(BaseConfig):
    """Configuration for ASR models with a pluggable encoder.

    Args:
        encoder_config: Encoder configuration instance, a dict, or None. If None,
            defaults to the predefined "zipformer-large" config. If a dict, it must
            contain a "model_type" key (defaults to "zipformer").
        decoder_dim: Decoder hidden dimension.
        context_size: Decoder context size (e.g., for transducer joiner context).
        joiner_dim: Joiner hidden dimension.
        use_transducer: Whether to enable transducer criterion.
        use_ctc: Whether to enable CTC criterion.
        **kwargs: Extra fields stored for forward compatibility.
    """

    model_type = "asr"

    def __init__(
        self,
        encoder_config=None,
        decoder_dim: int = 512,
        context_size: int = 2,
        joiner_dim: int = 512,
        use_transducer: bool = True,
        use_ctc: bool = False,
        **kwargs,
    ):
        if encoder_config is None:
            enc_cfg = ZipformerConfig()
        elif isinstance(encoder_config, dict):
            enc_type = encoder_config.get("model_type", "zipformer")
            if enc_type != "zipformer":
                raise ValueError(
                    "encoder_config.model_type must be 'zipformer' or pass a ZipformerConfig instance"
                )
            enc_kwargs = {k: v for k, v in encoder_config.items() if k != "model_type"}
            enc_cfg = ZipformerConfig(**enc_kwargs)
        else:
            enc_cfg = encoder_config

        self.encoder_config = enc_cfg
        self.decoder_dim = decoder_dim
        self.context_size = context_size
        self.joiner_dim = joiner_dim
        self.use_transducer = use_transducer
        self.use_ctc = use_ctc

        super().__init__(**kwargs)
