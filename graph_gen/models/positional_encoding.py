import numpy as np
import torch


class PositionalEncoding(torch.nn.Module):
    def __init__(self, dimension: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dimension, 2) * (-np.log(10000.0) / dimension))
        pe = torch.zeros(max_len, dimension)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self) -> torch.Tensor:
        return self.pe