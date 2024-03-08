import torch
import layers
import pos 

# ---- MODULES ----
K_MASK = -2.3819763e38  # Set to a large negative number.
LayerCache = dict[str, torch.Tensor]
def init_layer_cache(
    cache_size: int,
    num_heads: int,
    head_dim: int,
    batch_size: int,
    dtype: torch.dtype = torch.bfloat16,
    ) -> LayerCache:
    return {
        'v': torch.zeros(
            (batch_size, cache_size, num_heads, head_dim), dtype=dtype
        ),
        'k': torch.zeros(
            (batch_size, cache_size, num_heads, head_dim), dtype=dtype
        ),
        'end_index': torch.zeros((batch_size,), dtype=torch.int32),
    }


class Embedding(torch.nn.Module):
    vocab_size: int 
    embed_dim: int
    def setup(self):
            self.input_embedding_table = self.param(
            'input_embedding',
            torch.nn.init.normal_(),
            (self.vocab_size, self.embed_dim),
        )
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_embedding_table[(x,)]
        x *= torch.sqrt(self.embed_dim).astype(x.dtype)
        return x
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.dot(x, self.input_embedding_table.T)

class Attention(torch.nn.Module):
    num_heads: int
    num_kv_heads: int
    features: int
    head_dim: int
    LayerCache = dict[str, torch.Tensor]
    def __init__(self, num_heads, num_kv_heads, features, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.features = features
        self.head_dim = head_dim
    @property
    def use_qkv_einsum(self):
        return self.num_kv_heads == self.num_heads

    def setup(self):
        self.attn_vec_einsum = layers.Einsum(
            shape=(self.num_heads, self.head_dim, self.features),
        )

        if self.use_qkv_einsum:
            self.qkv_einsum = layers.Einsum(
                shape=(3, self.num_heads, self.features, self.head_dim),
            )
        else:
            self.q_einsum = layers.Einsum(
                shape=(self.num_heads, self.features, self.head_dim),
            )
            self.kv_einsum = layers.Einsum(
                shape=(2, self.num_kv_heads, self.features, self.head_dim),
            )
    def __call__(
        self,
        x: torch.Tensor,
        segment_pos: torch.Tensor,
        cache: LayerCache | None,
        attn_mask: torch.Tensor,
        ) -> tuple[LayerCache | None, torch.Tensor]:
        seq_len = x.shape[1]

        if self.use_qkv_einsum:
            query_proj, key_proj, value_proj = self.qkv_einsum('BTD,SNDH->SBTNH', x)
        else:
            query_proj = self.q_einsum('BTD,NDH->BTNH', x)
            key_proj, value_proj = self.kv_einsum('BSD,CKDH->CBSKH', x)

        query_proj = pos.add_rope(
            query_proj,
            segment_pos,
            head_dim=self.head_dim,
        )
        query_scaled = query_proj * self.head_dim**-0.5
        key_proj = pos.add_rope(
            key_proj,
            segment_pos,
            head_dim=self.head_dim,
        )

        if not self.use_qkv_einsum:
            value_proj = torch.Tensor.repeat(value_proj, self.num_heads, axis=-2)
            key_proj = torch.repeat(key_proj, self.num_heads, axis=-2)
        if cache is not None:
            end_index = cache['end_index'][0]
            slice_indices = (0, end_index % cache['v'].shape[1], 0, 0)
            value_proj = torch.index_put(
                cache['v'],
                value_proj,
                slice_indices,
            )
            key_proj = torch.index_put(
                cache['k'], key_proj, slice_indices
            )

        logits = torch.einsum('BTNH,BSNH->BTNS', query_scaled, key_proj)
        padded_logits = torch.where(
                torch.unsqueeze(attn_mask, dim=-2), logits, self.K_MASK
            ) 
        probs = torch.nn.Softmax(padded_logits, axis=-1).astype(key_proj.dtype)
        encoded = torch.einsum('BTNS,BSNH->BTNH', probs, value_proj)
        attn_output = self.attn_vec_einsum('BTNH,NHD->BTD', encoded)

        if cache is not None:
            new_cache = {
                'v': value_proj,
                'k': key_proj,
                'end_index': cache['end_index'] + seq_len,
            }
        else:
            new_cache = None

        return new_cache, attn_output
class FeedForward(torch.nn.Module):
      features: int
      hidden_dim: int
 
      def __call__(self, x):
        w_gating = self.param(
            'gating_einsum',
            torch.nn.init.zeros_(),
            ((2, self.features, self.hidden_dim)),
        )
        ff_gate = torch.dot(x, w_gating[0])
        gate_value = torch.nn.GELU(ff_gate)

        ff1 = torch.dot(x, w_gating[1])
        activations = gate_value * ff1

        w_linear = self.param(
            'linear',
            torch.nn.init.zeros_(),
            (self.hidden_dim, self.features),
        )
        outputs = torch.dot(activations, w_linear)

        return outputs


class Block(torch.nn.Module):

    num_heads: int
    num_kv_heads: int
    embed_dim: int
    head_dim: int
    hidden_dim: int

    def setup(self):
        self.pre_attention_norm = layers.RMSNorm()
        self.attn = Attention(
            num_heads=self.num_heads,
            features=self.embed_dim,
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
        )
        self.pre_ffw_norm = layers.RMSNorm()
        self.mlp = FeedForward(features=self.embed_dim, hidden_dim=self.hidden_dim)

    def __call__(
        self,
        x: torch.Tensor,
        segment_pos: torch.Tensor,
        cache: LayerCache | None,
        attn_mask: torch.Tensor,
    ) -> tuple[LayerCache | None, torch.Tensor]:
        inputs_normalized = self.pre_attention_norm(x)
        cache, attn_output = self.attn(
            inputs_normalized,
            segment_pos,
            cache,
            attn_mask,
        )
        attn_output += x
        residual = attn_output
        attn_output = self.pre_ffw_norm(attn_output)
        outputs = self.mlp(attn_output)
        outputs = residual + outputs
        return cache, outputs