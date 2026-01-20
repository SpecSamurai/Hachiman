from typing import Any
import torch
import tiktoken
from .gptModel import GPTModel

def text_to_ids(text: str, tokenizer: tiktoken.Encoding):
    encoded_text = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded_text).unsqueeze(0)
    return encoded_tensor

def ids_to_text(idx: torch.Tensor, tokenizer: tiktoken.Encoding):
    print('ids_to_text')
    print(' ids Type: ', type(idx))
    print(' IDS shape:', idx.shape)
    flat = idx.squeeze(0)
    return tokenizer.decode(flat.tolist())

def generate_text(model: GPTModel, idx: torch.Tensor, max_new_tokens: int, context_size: Any):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def generate(model: GPTModel, idx: torch.Tensor, max_new_tokens: int, context_size: int, temperature:float = 0.0, topK = None, eos_id = None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if topK is not None:
            top_logits, _ = torch.topk(logits, k=topK)
            min_val = top_logits[:, -1]
            logits = torch.where(
                condition=logits < min_val,
                input=torch.tensor(float('-inf')).to(logits.device),
                other=logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            probas = torch.softmax(logits, dim=1)
            idx_next = torch.multinomial(probas, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_cond == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)
    return idx

