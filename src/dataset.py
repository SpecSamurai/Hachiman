# import re
import torch, tiktoken

from typing import Any, override
from torch.utils.data import Dataset, DataLoader

# def getVocab(filePath: str) -> dict[str, int]:
#     with open(filePath, 'r', encoding='utf-8') as f:
#         raw_text = f.read()
#
#     preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
#     preprocessed = [item for item in preprocessed if item.strip()]
#
#     tokens = sorted(set(preprocessed))
#     tokens.extend(["<|endoftext|>", "<|unk|>"])
#     vocab = {token:integer for integer,token in enumerate(tokens)}
#
#     return vocab

# class SimpleTokenizerV1:
#     strToInt: dict[str, int]
#     intToStr: dict[int, str]
#
#     def __init__(self, vocab: dict[str, int]) -> None:
#         self.strToInt = vocab
#         self.intToStr = {i:s for s,i in vocab.items()}
#
#     def encode(self, text: str):
#         preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
#         preprocessed = [item.strip() for item in preprocessed if item.strip()]
#         preprocessed = [item if item in self.strToInt
#                         else "<|unk|>" for item in preprocessed]
#
#         ids = [self.strToInt[s] for s in preprocessed]
#         return ids
#
#
#     def decode(self, ids: list[int]):
#         text = " ".join([self.intToStr[i] for i in ids])
#         text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
#         return text

# vocab = getVocab('/home/spec/Code/Hachiman/the-verdict.txt')
# tokenizer = SimpleTokenizerV1(vocab)
# ids = tokenizer.encode("Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace.")
# # ids = tokenizer.encode("Hello world")
# print(ids)
# print(tokenizer.decode(ids))

class GPTDatasetV1(Dataset[Any]):
    inputIds: list[torch.Tensor]
    targetIds: list[torch.Tensor]

    def __init__(self, text: str, tokenizer: tiktoken.Encoding, maxLength: int, stride: int) -> None:
        self.inputIds = []
        self.targetIds = []

        tokenIds = tokenizer.encode(text)

        for i in range(0, len(tokenIds) - maxLength, stride):
            inputChunk = tokenIds[i:i + maxLength]
            targetChunk = tokenIds[i + 1:i + maxLength + 1]

            self.inputIds.append(torch.Tensor(inputChunk))
            self.targetIds.append(torch.Tensor(targetChunk))

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

    return raw_text

dataloader = createDataLoaderV1(loadTestDocu(), batch_size=8, maxLength=4, stride=4, shuffle=False)
dataIter = iter(dataloader)

torch.set_printoptions(sci_mode=False)
print(next(dataIter))
