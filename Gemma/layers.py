import torch

class RMS(torch.nn.Module):
      def __call__(self, x):
        scale = self.param('scale', torch.nn.init.zeros_(), (x.shape[-1]))
        var = torch.mean(torch.square(x),keepdim=True, axis=-1)
        normed_inputs = torch.asarray(x * torch.reciprocal(torch.square(var + 1e-06)))
        normed_inputs = normed_inputs * (1 + scale)
        return normed_inputs
class Einsum(torch.nn.Module):
        def __init__(self, shape: tuple[int, ...]):
            super().__init__()
            self.shape = shape
            self.w = torch.nn.Parameter(torch.empty(*shape))
            torch.nn.init.normal_(self.w)

        def forward(self, eqn: str, x: torch.Tensor) -> torch.Tensor:
            return torch.einsum(eqn, x, self.w)