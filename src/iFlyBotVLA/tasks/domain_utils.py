import torch
from torch import nn


class DomainAwareLinear(nn.Module):
    """
    Linear layer with domain-conditioned parameters (per-sample).

    Each domain has its own weight and bias vectors, stored in embeddings.
    """

    def __init__(self, input_size: int, output_size: int, num_domains: int = 30) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Embedding(num_domains, output_size * input_size)
        self.bias = nn.Embedding(num_domains, output_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: torch.Tensor, domain_id: torch.LongTensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            [B, I] or [B, T, I]
        domain_id : LongTensor
            [B], domain indices.

        Returns
        -------
        Tensor
            [B, O] or [B, T, O]
        """
        B = domain_id.shape[0]
        squeeze_T = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_T = True
        W = self.fc(domain_id).view(B, self.input_size, self.output_size)
        b = self.bias(domain_id).view(B, self.output_size)
        y = torch.matmul(x, W) + b.view(B, 1, self.output_size)
        if squeeze_T:
            y = y.squeeze(1)
        return y
    