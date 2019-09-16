import torch
from loaders.vqa import *
from random import *
from pytorch_transformers import GPT2Tokenizer


model = torch.load('model_checkpoint.h5')
dataset = VQADataset()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<PAD>')

sample = dataset[458]

dataset.decode_sample(sample)

outputs = model(sample[0].unsqueeze(0), imgages=sample[1].unsqueeze(0))

preds = outputs[0]

for pred in preds:
    print(tokenizer.decode(torch.argmax(pred).item()))
