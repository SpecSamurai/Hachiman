import torch, tiktoken

from typing import Any, override
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset[Any]):
    inputIds: list[torch.Tensor]
    targetIds: list[torch.Tensor]

    def __init__(
            self,
            text: str,
            tokenizer: tiktoken.Encoding,
            maxLength: int,
            stride: int) -> None:
        self.inputIds = []
        self.targetIds = []

        tokenIds = tokenizer.encode(text)
        print('Tokens\n', len(tokenIds))

        for i in range(0, len(tokenIds) - maxLength, stride):
            inputChunk = tokenIds[i:i + maxLength]
            targetChunk = tokenIds[i + 1:i + maxLength + 1]

            self.inputIds.append(torch.tensor(inputChunk))
            self.targetIds.append(torch.tensor(targetChunk))

        print('Created Dataset')

    def __len__(self) -> int:
        return len(self.inputIds)

    @override
    def __getitem__(self, index: int) -> Any:
        return self.inputIds[index], self.targetIds[index]

def createDataLoaderV1(
    text: str,
    batch_size: int = 4,
    maxLength: int = 256,
    stride: int = 126,
    shuffle: bool =  True,
    drop_last: bool = True,
    num_worker: int = 0) -> DataLoader[Any]:

    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDatasetV1(text, tokenizer, maxLength, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_worker
    )

    print('Created DataLoader')

    return dataloader

def loadTestDocu() -> str:
    with open('/home/spec/Code/Hachiman/the-verdict.txt', 'r', encoding='utf-8') as f:
        raw_text = f.read()

    print('Loaded test document')
    print('Characters ', len(raw_text))

    return raw_text
