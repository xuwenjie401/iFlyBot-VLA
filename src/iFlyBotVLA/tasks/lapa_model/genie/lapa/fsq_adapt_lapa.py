## https://github.com/MHVali/Noise-Substitution-in-Vector-Quantization/blob/main/NSVQ.py
## NSVQ: Noise Substitution in Vector Quantization for Machine Learning in IEEE Access journal, January 2022

import torch
import torch.distributions.normal as normal_dist
import torch.distributions.uniform as uniform_dist
from typing import List, Optional
import torch.nn as nn
from torch import Tensor, int32
## add project_in, project_out layer 
## FYI vector_quantize_pytorch
class FSQ(torch.nn.Module):
    def __init__(self, levels, eps=1e-3, dim=16, device=torch.device('cpu'), code_seq_len=4, patch_size=32, image_size = 256):
        super(FSQ, self).__init__()

        """
        Inputs:
        
        1. num_embeddings = Number of codebook entries
        
        2. embedding_dim = Embedding dimension (dimensionality of each input data sample or codebook entry)
        
        3. device = The device which executes the code (CPU or GPU)
        
        ########## change the following inputs based on your application ##########
        
        4. discarding_threshold = Percentage threshold for discarding unused codebooks
        
        5. initialization = Initial distribution for codebooks

        """
        self._levels = levels
        self._eps = eps
        self._dim = dim
        self._levels_tensor = torch.tensor(levels, dtype=int32).to(device)
        self._basis = torch.cumprod(torch.tensor([1] + self._levels[:-1]), dim=0, dtype=int32).to(device)
        self._implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size, device=device))

        self.image_size = image_size
        self.device = device
        self.patch_size = patch_size

        
        self.project_in = torch.nn.Linear(dim, 32)
        self.project_out = torch.nn.Sequential(
            torch.nn.Linear(len(levels), 32), 
            torch.nn.GELU(),
            torch.nn.Linear(32, dim))
        if code_seq_len == 4:
            self.cnn_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
                # torch.nn.LayerNorm([16, 4, 4]),
                torch.nn.GELU(),
                torch.nn.Conv2d(in_channels=16, out_channels=len(levels), kernel_size=3, stride=1, padding=0),
                torch.nn.LayerNorm([len(levels), 2, 2]),
            )
        elif code_seq_len == 1:
            self.cnn_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
                # torch.nn.LayerNorm([16, 4, 4]),
                torch.nn.GELU(),
                torch.nn.Conv2d(in_channels=16, out_channels=len(levels), kernel_size=4, stride=1, padding=0),
                torch.nn.LayerNorm([len(levels), 1, 1]),
            )
        

    def encode(self, input_data, batch_size):
        # compute the distances between input and codebooks vectors
        input_data = self.project_in(input_data) # b * 64 * 32
        # change the order of the input_data to b * 32 * 64
        input_data = input_data.permute(0, 2, 1).contiguous()
        # reshape input_data to 4D b*h*w*d
        input_data = input_data.reshape(batch_size, 32, int(self.image_size/self.patch_size), int(self.image_size/self.patch_size))
        # import pdb
        # pdb.set_trace()
        input_data = self.cnn_encoder(input_data) # 1*1 tensor
        input_data = input_data.reshape(batch_size, len(self._levels), -1) # b * 32 * d^2
        input_data = input_data.permute(0, 2, 1).contiguous() # b * 1 * 32
        input_data = input_data.reshape(-1, len(self._levels))
        return input_data
    
    def decode(self, quantized_input, batch_size):
        # import pdb
        # pdb.set_trace()
        quantized_input = quantized_input.reshape(batch_size, len(self._levels), -1) # b * 32 * d^2
        quantized_input = quantized_input.permute(0, 2, 1).contiguous() # b * 64 * 32
        
        quantized_input = self.project_out(quantized_input)
        return quantized_input
    
    def forward(self, input_data_first, input_data_last):

        """
        This function performs the main proposed vector quantization function using NSVQ trick to pass the gradients.
        Use this forward function for training phase.

        N: number of input data samples
        K: num_embeddings (number of codebook entries)
        D: embedding_dim (dimensionality of each input data sample or codebook entry)

        input: input_data (input data matrix which is going to be vector quantized | shape: (NxD) )
        outputs:
                quantized_input (vector quantized version of input data used for training | shape: (NxD) )
                perplexity (average usage of codebook entries)
        """
        batch_size = input_data_first.shape[0]
        # import pdb
        # pdb.set_trace()
        input_data_first = input_data_first.contiguous()

        input_data_first = self.encode(input_data_first, batch_size) # b * 1 * 32
        input_data_last = self.encode(input_data_last, batch_size) # b * 1 * 32
        
        input_data = input_data_last - input_data_first
        codes = self.quantize(input_data)
        indices = self.codes_to_indices(codes)
        quantized_input = self.decode(codes, batch_size)
        return quantized_input, indices.reshape(batch_size, -1)

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

    def inference(self, input_data_first, input_data_last, user_action_token_num=None):

        """
        This function performs the vector quantization function for inference (evaluation) time (after training).
        This function should not be used during training.

        N: number of input data samples
        K: num_embeddings (number of codebook entries)
        D: embedding_dim (dimensionality of each input data sample or codebook entry)

        input: input_data (input data matrix which is going to be vector quantized | shape: (NxD) )
        outputs:
                quantized_input (vector quantized version of input data used for inference (evaluation) | shape: (NxD) )
        """

        #input_data = self.project_in(input_data)
        input_data_first = input_data_first.detach().clone()
        input_data_last = input_data_last.detach().clone()
        ###########################################
        
        batch_size = input_data_first.shape[0]
        # compute the distances between input and codebooks vectors
        
        input_data_first = self.encode(input_data_first, batch_size) # b * n * dim
        input_data_last = self.encode(input_data_last, batch_size) # b * n * dim
        
        input_data = input_data_last - input_data_first
        input_data = input_data.reshape(-1, len(self._levels))     

        codes = self.quantize(input_data)
        indices = self.codes_to_indices(codes)
        quantized_input = self.decode(codes, batch_size)

        #use the tensor "quantized_input" as vector quantized version of your input data for inference (evaluation) phase.
        return quantized_input, indices.reshape(batch_size, -1), codes
        