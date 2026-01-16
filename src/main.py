import torch
import tiktoken

from .gptModel import GPTModel
from .dataset import createDataLoaderV1, loadTestDocu

VOCAB_SIZE_STR: str = 'vocab_size'
CONTEXT_LENGTH_STR: str = 'context_length'
EMB_DIM_STR: str = 'emb_dim'
N_HEADS_STR: str = 'n_heads'
N_LAYERS_STR: str = 'n_layers'
DROP_RATE_STR: str = 'drop_rate'
QKV_BIAS_STR: str = 'qkv_bias'

GPT_CONFIG_124M = {
    VOCAB_SIZE_STR: 50257,
    CONTEXT_LENGTH_STR: 1024,
    EMB_DIM_STR: 768,
    N_HEADS_STR: 12,
    N_LAYERS_STR: 12,
    DROP_RATE_STR: 0.1,
    QKV_BIAS_STR: False,
}

def generate_text(model: GPTModel, idx: torch.Tensor, max_new_tokens: int, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Support: {torch.cuda.is_available()}')

torch.set_printoptions(sci_mode=False)

dataloader = createDataLoaderV1(loadTestDocu(), batch_size=8, maxLength=4, stride=4, shuffle=False)
dataIter = iter(dataloader)
# print(next(dataIter))

tokenizer = tiktoken.get_encoding('gpt2')
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch.shape)

torch.manual_seed(123)

model = GPTModel(GPT_CONFIG_124M)
logits = model(batch)
print(logits.shape)
print(logits)

start_context = 'Hello, I am'
encoded = tokenizer.encode(start_context)
print('encoded ', encoded)

encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print(encoded_tensor.shape)

model.eval()
out = generate_text(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M[CONTEXT_LENGTH_STR]
)

print('output ', out)
print(out.shape)

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)

