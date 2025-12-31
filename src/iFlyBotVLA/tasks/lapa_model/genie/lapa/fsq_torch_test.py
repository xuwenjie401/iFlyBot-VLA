import torch
import numpy as np
from typing import List, Optional
import torch.nn as nn
from torch import Tensor, int32
import pdb
class FSQ(nn.Module):
    """Quantizer (PyTorch version)."""
    def __init__(self, levels: List[int], eps: float = 1e-3, dim: Optional[int] = None):
        super().__init__()
        self._levels = levels
        self._eps = eps
        self._dim = 16
        self._levels_tensor = torch.tensor(levels, dtype=int32)
        self._basis = torch.cumprod(torch.tensor([1] + self._levels[:-1]), dim=0, dtype=int32)
        self._implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size))
        self.project_in = nn.Linear(self._dim, len(self._levels))
        self.project_out = nn.Linear(len(self._levels), self._dim)

    @property
    def num_dimensions(self) -> int:
        """Number of dimensions expected from inputs."""
        return len(self._levels)

    @property
    def codebook_size(self) -> torch.Tensor:
        """Size of the codebook."""
        return self._levels_tensor.prod().item()
    @property
    def codebook(self) -> torch.Tensor:
        """Returns the implicit codebook. Shape (prod(levels), num_dimensions)."""
        return self._implicit_codebook

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels_tensor - 1) * (1 - self._eps) / 2
        offset = torch.where(self._levels_tensor % 2 == 1, 0.0, 0.5)
        shift = torch.tan(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = self.round_ste(self.bound(z))
        
        # Renormalize to [-1, 1]
        half_width = self._levels_tensor // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        # Scale and shift to range [0, ..., L-1]
        half_width = self._levels_tensor // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
        half_width = self._levels_tensor // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: torch.Tensor) -> torch.Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.num_dimensions
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(torch.int32)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Inverse of `indexes_to_codes`."""
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels_tensor
        return self._scale_and_shift_inverse(codes_non_centered)

    def round_ste(self, z: torch.Tensor) -> torch.Tensor:
        """Round with straight through gradients."""
        zhat = torch.round(z)
        return z + (zhat - z).detach()

    def forward(self, z: Tensor) -> Tensor:
        z = self.project_in(z)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        out = self.project_out(codes)

        return out, indices

if __name__ == '__main__':
    levels = [8,6,5] 
    fsq = FSQ(levels)
    x = torch.randn(1, 4, 4, 16) 
    pdb.set_trace()
    xhat, indices = fsq(x)
    codes = fsq.indices_to_codes(indices)
    for name, param in fsq.named_parameters():
        if param.requires_grad: 
            print(f"名称: {name}, 参数: {param}")