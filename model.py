import math
import os

import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import _LRScheduler

device = torch.device("cuda")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model,
            nhead,
            d_hid,
            dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, nlayers, norm=RMSNorm(d_model)
        )
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.dummy_param = nn.Parameter(torch.empty(0))

        self.init_weights()

    @property
    def device(self):
        return self.dummy_param.device

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[batch_size, seq_len, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = (
                nn.Transformer.generate_square_subsequent_mask(src.shape[1])
                .to(self.device)
                .isinf()
            )
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output

    @torch.inference_mode()
    def inference(self, prefix: str = "", temp: float = 1.0) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()

        indices = [self.dataset.bos_id] + self.dataset.text2ids(prefix)
        indices = torch.tensor(indices).unsqueeze(0).to(next(self.parameters()).device)

        embeds = self.embedding(indices)
        output, hidden = self.rnn(embeds)
        logits = self.linear(output) / temp

        new_indexes = Categorical(logits=logits[:, -1:]).sample()
        indices = torch.cat([indices, new_indexes], dim=1)

        # print(f'{indices=}')
        # print(f'{self.dataset.ids2text(indices)=}')

        while indices.shape[1] < self.max_length:
            if new_indexes.item() == self.dataset.eos_id:
                break

            embeds = self.embedding(indices)
            logits = self.forward(embeds, hidden)

            new_indexes = Categorical(logits=logits[:, -1:]).sample()
            indices = torch.cat([indices, new_indexes], dim=1)
            # print(f'{indices=}')
            # print(f'{self.dataset.ids2text(indices)=}')

        return self.dataset.ids2text(indices.squeeze()[1:])


class WarmUpScheduler(_LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1, min_lr=0.0):
        self.min_lr = min_lr
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch)
        return [
            max(
                self.min_lr,
                base_lr
                * self.d_model ** (-0.5)
                * min(
                    step ** (-0.5),
                    step * self.warmup_steps ** (-1.5),
                ),
            )
            for base_lr in self.base_lrs
        ]


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        """
        Root Mean Square Layer Normalization
        :param d_model: model size
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d_model = d_model

        self.scale = nn.Parameter(torch.ones(d_model))
        self.register_parameter("scale", self.scale)

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)

        rms_x = norm_x / torch.sqrt(torch.tensor(self.d_model))
        x_normed = x / (rms_x + self.eps)

        return self.scale * x_normed
