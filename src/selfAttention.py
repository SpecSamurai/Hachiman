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

class MultiHeadAttention(torch.nn.Module):
    d_out: int
    num_heads: int
    head_dim: int
    W_query: torch.nn.Linear
    W_key: torch.nn.Linear
    W_value: torch.nn.Linear
    out_proj: torch.nn.Linear
    dropout: torch.nn.Dropout

    def __init__(
            self,
            d_in: int,
            d_out: int,
            context_length: int,
            dropout: float,
            num_heads: int,
            qkv_bias: bool = False) -> None:
        super().__init__()

        assert(d_out % num_heads == 0), 'd_out must be divisable by num_heads'
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_out, d_out)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        print(keys)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        print(keys)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        print(keys)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attentionScores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attentionScores.masked_fill_(mask_bool, -torch.inf)
        attentionWeights = torch.softmax(attentionScores / keys.shape[-1]**0.5, dim=-1)
        attentionWeights = self.dropout(attentionWeights)

        context_vector = (attentionWeights @ values).transpose(1, 2)
        context_vector = context_vector.contiguous().view(
            b, num_tokens, self.d_out
        )

        context_vector = self.out_proj(context_vector)

        return context_vector
