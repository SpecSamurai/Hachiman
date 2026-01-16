from typing import override
import torch

from typing_extensions import Any
from .selfAttention import MultiHeadAttention
from .activations import GELU
from .norms import LayerNorm

class GPTModel(torch.nn.Module):
    tok_emb: torch.nn.Embedding
    pos_emb: torch.nn.Embedding
    drop_emb: torch.nn.Dropout
    trf_blocks: torch.nn.Sequential
    final_norm: LayerNorm
    out_head: torch.nn.Linear

    def __init__(self, cfg: dict[Any, Any]) -> None:
        super().__init__()
        self.tok_emb = torch.nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = torch.nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = torch.nn.Dropout(cfg['drop_rate'])
        self.trf_blocks = torch.nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = torch.nn.Linear(
            cfg['emb_dim'], cfg['vocab_size'], bias=False
        )

    @override
    def forward(self, in_idx: torch.Tensor):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)
        return logits

class FeedForward(torch.nn.Module):
    layers: torch.nn.Sequential

    def __init__(self, cfg: dict[Any, Any]):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            torch.nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']),
        )

    @override
    def forward(self, x: torch.Tensor):
        return self.layers(x)

class TransformerBlock(torch.nn.Module):
    mha: MultiHeadAttention
    norm1: LayerNorm
    norm2: LayerNorm
    ff: FeedForward
    drop_shortcut: torch.nn.Dropout
    
    def __init__(self, cfg: dict[Any, Any]):
        super().__init__()
        self.mha = MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            dropout=cfg['drop_rate'],
            num_heads=cfg['n_heads'],
            qkv_bias=cfg['qkv_bias']
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = torch.nn.Dropout(cfg['drop_rate'])

    @override
    def forward(self, x: torch.Tensor):

        shortcut = x
        x = self.norm1(x)
        x = self.mha(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x

