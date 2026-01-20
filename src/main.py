import tiktoken
import torch

from .gptModel import GPTModel
from .dataset import createDataLoaderV1, loadTestDocu
from .print import generate, generate_text, text_to_ids, ids_to_text

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

def softmax_with_temperatur(logits: torch.Tensor, temperature: float):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor, model: GPTModel, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:

            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )

            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches

def evaluate_model(model, train_loader, validate_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(validate_loader, model, device, num_batches=eval_iter)

    model.train()

    return train_loss, val_loss

def generate_and_print(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )

    decoded_text = ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace('\n', ' '))

    model.train()

def train_model(
    model, 
    train_loader, 
    validate_loader, 
    optimizer, 
    device, 
    num_epochs, 
    eval_freq, 
    eval_iter, 
    start_context, 
    tokenizer):
    train_losses, val_losses, track_token_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for (input_batch, target_batch) in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )

            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, validate_loader, device, eval_iter
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(tokens_seen)

                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

        generate_and_print(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_token_seen

print(f'PyTorch Version: {torch.__version__}')

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

torch.set_printoptions(sci_mode=False)

text_data = loadTestDocu()

train_ratio = 0.9
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
validate_data = text_data[split_idx:]

train_loader = createDataLoaderV1(
    train_data,
    batch_size=2, # small, in practice 1024 is more likely
    maxLength=GPT_CONFIG_124M[CONTEXT_LENGTH_STR],
    stride=GPT_CONFIG_124M[CONTEXT_LENGTH_STR],
    drop_last=True,
    shuffle=True,
    num_worker=0
)

validate_loader = createDataLoaderV1(
    validate_data,
    batch_size=2,
    maxLength=GPT_CONFIG_124M[CONTEXT_LENGTH_STR],
    stride=GPT_CONFIG_124M[CONTEXT_LENGTH_STR],
    drop_last=False,
    shuffle=False,
    num_worker=0
)

print('Train loader')
for x, y in train_loader:
    print(x.shape, y.shape)

print('Validation loader')
for x, y in validate_loader:
    print(x.shape, y.shape)

tokenizer = tiktoken.get_encoding('gpt2')
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 0.0004, weight_decay=0.1
)

num_epochs = 10

train_losses, validate_losses, tokens_seen = train_model(
    model, train_loader, validate_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5, 
    start_context='Every effort moves you', tokenizer=tokenizer
)

model.eval()
torch.manual_seed(123)
token_ids = generate(
    model=model,
    idx=text_to_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M[CONTEXT_LENGTH_STR],
    topK=25,
    temperature=1.4
)

print("Output text:\n", ids_to_text(token_ids, tokenizer))

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, 'model_and_optimizer.pth')

checkpoint = torch.load('model_and_optimizer.pth', map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.train()

model.eval()

token_ids = generate(
    model=model,
    idx=text_to_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M[CONTEXT_LENGTH_STR],
    topK=25,
    temperature=1.4
)
print("Output text:\n", ids_to_text(token_ids, tokenizer))


