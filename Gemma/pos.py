import torch

# ---- POS embedding ----
_MAX_WAVELENGTH = 10000
def add_pos_embedding(input_embedding: torch.Tensor, 
                      position: int, 
                      max_wavelength: int = _MAX_WAVELENGTH,) -> torch.Tensor:
  embed_dim = input_embedding[-1].shape
  num_timescales = embed_dim // 2
  log_timescale_increment = torch.log(float(max_wavelength)) / torch.maximum(
    torch.asarray(num_timescales, dtype=torch.float32) - 1, 1
  )
  inv_timescales = torch.exp(
      torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
  )
  scaled_time = position * inv_timescales
  signal = torch.concatenate([torch.sin(scaled_time), torch.cos(scaled_time)])
  signal = torch.pad(signal, [[0, torch.mod(embed_dim, 2)]])
  position_embedding = signal.astype(torch.float32)
  return input_embedding + position_embedding

# ---- RoPE Embeddings ----

def add_rope(inputs: torch.Tensor,    
            positions: torch.Tensor, 
            head_dim: int,
            max_wavelength: int = _MAX_WAVELENGTH,) -> torch.Tensor:
  fraction = 2 * torch.arange(0, head_dim // 2) / head_dim
  timescale = max_wavelength**fraction

  sinusoid_inp = (
      positions.unsqueeze(-1) / timescale.unsqueeze(-2)
  )
  sin = torch.sin(sinusoid_inp)
  cos = torch.cos(sinusoid_inp)

  first_half, second_half = torch.chunk(inputs, 2, dim=-1)
  first_part = first_half * cos - second_half * sin
  second_part = second_half * cos + first_half * sin
  out = torch.cat([first_part, second_part], dim=-1)
  return out.type(inputs.dtype)