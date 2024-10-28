import torch
import torch.nn as nn
from typing import Optional

class LSTM(nn.RNNBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ) -> None:
        super().__init__(
            mode='LSTM',
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size,
            device=device,
            dtype=dtype,
        )
