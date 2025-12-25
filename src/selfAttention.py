from typing import override
import torch

class SelfAttentionV2(torch.nn.Module):
    queryW: torch.nn.Linear
    keyW: torch.nn.Linear
    valueW: torch.nn.Linear

    def __init__(self, dIn: int, dOut: int, qkvBias: bool = False) -> None:
        super().__init__()
        # self.queryW = torch.nn.Parameter(torch.rand(dIn, dOut))
        # self.keyW = torch.nn.Parameter(torch.rand(dIn, dOut))
        # self.valueW = torch.nn.Parameter(torch.rand(dIn, dOut))
        self.queryW = torch.nn.Linear(dIn, dOut, bias=qkvBias)
        self.keyW = torch.nn.Linear(dIn, dOut, bias=qkvBias)
        self.valueW = torch.nn.Linear(dIn, dOut, bias=qkvBias)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keys = self.keyW(x)
        values = self.valueW(x)
        queries = self.queryW(x)

        attentionScores = queries @ keys.T
        attentionWeights = torch.softmax(
            attentionScores / keys.shape[-1]**0.5,
            dim=-1
        )

        contextVector = attentionWeights @ values
        return contextVector

    # @override
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     keys = x @ self.keyW
    #     print(keys.shape)
    #
    #     values = x @ self.valueW
    #     queries = x @ self.queryW
    #
    #     attentionScores = queries @ keys.T
    #     attentionWeights = torch.softmax(
    #         attentionScores / keys.shape[-1]**0.5,
    #         dim=-1
    #     )
    #
    #     contextVector = attentionWeights @ values
    #     return contextVector


class CasualAttention(torch.nn.Module):
    queryW: torch.nn.Linear
    keyW: torch.nn.Linear
    valueW: torch.nn.Linear
    dropout: torch.nn.Dropout
    dOut: int

    def __init__(
        self,
        dIn: int,
        dOut: int,
        contextLength: int,
        dropout: float,
        qkvBias: bool = False) -> None:

        super().__init__()

        self.dOut = dOut
        self.queryW = torch.nn.Linear(dIn, dOut, bias=qkvBias)
        self.keyW = torch.nn.Linear(dIn, dOut, bias=qkvBias)
        self.valueW = torch.nn.Linear(dIn, dOut, bias=qkvBias)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(contextLength, contextLength), diagonal=1)
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batches, numTokens, dIn = x.shape

        keys = self.keyW(x)
        values = self.valueW(x)
        queries = self.queryW(x)

        attentionScores = queries @ keys.transpose(1, 2)
        attentionScores.masked_fill_(
            self.mask.bool()[:numTokens, :numTokens], -torch.inf)
        attentionWeights = torch.softmax(
            attentionScores / keys.shape[-1]**0.5,
            dim=-1
        )

        attentionWeights = self.dropout(attentionWeights)

        print(attentionWeights)

        contextVector = attentionWeights @ values
        return contextVector
