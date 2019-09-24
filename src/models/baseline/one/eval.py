from helpers.dataset import *
from random import *
from pytorch_transformers import GPT2Tokenizer


model = torch.load('model_checkpoint.h5')
dataset = VQADataset()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<PAD>')

for _ in range(10):
    idx = randint(0, 100000)
    sample = dataset[idx]

    print(dataset.decode_sample(dataset.samples[idx]))

    outputs = model(sample[0].unsqueeze(0).to('cuda'), images=sample[1].unsqueeze(0).to('cuda'))

    preds = outputs[0][0]
    out = ''
    for pred in preds:
        out += tokenizer.decode(torch.argmax(pred).item())
    print(out)
    print('*' * 100)
