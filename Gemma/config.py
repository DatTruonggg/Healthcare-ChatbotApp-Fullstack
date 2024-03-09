import dataclasses
import immutabledict
import torch
from typing import Optional


# Keep a mapping from dtype strings to the supported torch dtypes.
_STR_DTYPE_TO_TORCH_DTYPE = immutabledict.immutabledict({
    'float16': torch.float16,
    'float': torch.float32,
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
})


@dataclasses.dataclass
class GemmaConfig:
    vocab_size: int = 256000
    max_position_embeddings: int = 8192
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    hidden_size: int = 3072
    intermediate_size: int = 24576
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    dtype: str = 'bfloat16'
    quant: bool = False
    tokenizer: Optional[str] = 'tokenizer/tokenizer.model'

    def get_dtype(self) -> Optional[torch.dtype]:
        return _STR_DTYPE_TO_TORCH_DTYPE.get(self.dtype, None)

def get_config_for_2b() -> GemmaConfig:
    return GemmaConfig(
        num_hidden_layers=18,
        num_attention_heads=8,
        num_key_value_heads=1,
        hidden_size=2048,
        intermediate_size=16384
    )


def get_model_config(variant: str) -> GemmaConfig:
    if variant == '2b':
        return get_config_for_2b()
    return ValueError(f'Invalid variant {variant}. Supported variants are "2b"'
                      'and "7b"')