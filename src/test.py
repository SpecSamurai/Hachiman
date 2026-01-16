import torch
import tiktoken
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.autograd import grad
from typing import override
from torch.utils.data import Dataset, DataLoader
from typing import Any
from .selfAttention import MultiHeadAttention
from .gptModel import GPTModel, LayerNorm, GELU, FeedForward

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
z = x1 * w1 + b
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)

print(loss)

grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)

print(grad_L_w1)
print(grad_L_b)

loss.backward()

print(w1.grad)
print(b.grad)


class NeuralNetwork(torch.nn.Module):
    layers: torch.nn.Sequential

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()

        self.layers = torch.nn.Sequential(
                # 1st hidden layer
                torch.nn.Linear(num_inputs, 30),
                torch.nn.ReLU(),

                # 2md hiddem layer
                torch.nn.Linear(30, 20),
                torch.nn.ReLU(),

                # output
                torch.nn.Linear(20, num_outputs),
                )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.layers(x)
        return logits

class ToyDataset(Dataset[Any]):
    features: torch.Tensor
    labels: torch.Tensor

    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.features = X
        self.labels = y

    @override
    def __getitem__(self, index: int):
        x = self.features[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return self.labels.shape[0]

torch.manual_seed(123)
model = NeuralNetwork(50, 3)
print(model)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total number of params:', num_params)

X = torch.rand((1, 50))
print(X)

out = model(X)
print(out)

with torch.no_grad():
    out: torch.Tensor = model(X)

print(out)

with torch.no_grad():
    out = torch.softmax(model(X), dim=1)

print(out)


X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5],
    ])

y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6]
    ])

y_test = torch.tensor([0, 1])

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        drop_last=True
        )

test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=False
        )

for idx, (x, y) in enumerate(train_loader):
    print(f'Batch {idx}', x, y)

M = NeuralNetwork(2, 2)
optimizer = torch.optim.SGD(M.parameters(), lr=0.5)

num_epochs = 3
for epoch in range(num_epochs):

    M.train()

    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = M(features)

        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch: {epoch}'
              f'Batch {batch_idx}'
              f'Train loss {loss}')

    M.eval()

with torch.no_grad():
    output = M(X_train)

print(output)

torch.set_printoptions(sci_mode=False)
# probas = torch.softmax(output, dim=1)
# print(probas)

tokenizer = tiktoken.get_encoding('gpt2')

text = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace. asdasdas"
        )
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)

inputs = torch.Tensor([[1, 2, 3], [4, 5, 6],[1, 2, 3], [4, 5, 6],[1, 2, 3], [4, 5, 6]])

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)

casualAttention = MultiHeadAttention(3, 2, 6, 0.5, 2)
context_vecs = casualAttention(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

ln = LayerNorm(5)
batch_e = torch.randn(2, 5)
out_ln = ln(batch_e)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, keepdim=True, unbiased=False)
print(mean)
print(var)

gelu, relu = GELU(), torch.nn.ReLU()
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))

for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)

plt.tight_layout()
plt.savefig('plot.png')
plt.close()

# ffn = FeedForward(GPT_CONFIG_124M)
# x = torch.rand(2, 3, 768)
# out = ffn(x)
# print(out.shape)
