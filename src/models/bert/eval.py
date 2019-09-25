import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)
sys.path.append('/home/alberto/PycharmProjects/BlindLess/data/vqa')

print(root_path)

import torch
import random
from models.bert import model
from helpers.dataset import *
from pytorch_transformers import BertTokenizer


# Load the model
model = torch.load('checkpoints/bert_vgg_DS%_0.5_B_64_LR_5e-05_CHKP_EPOCH_2.h5')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
decode = lambda text: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

# Load some data (seed to replicate experiments)
seed = random.seed(867)
dataset = VQADataset(fname='bert_vgg_padded_types_dataset.pk')
idx = random.randint(0, len(dataset))

print(dataset.samples[idx])

# Hardware
device = torch.cuda.current_device()
model.to(device)


image = dataset[idx][4].unsqueeze(0).to(device)
answer = dataset.samples[idx].answer

while True:

    question = input('Question: ')
    question = '[CLS] ' + question + ' [SEP]'
    _input = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(question))
    input_type_ids = [1] * len(_input)
    att_mask = [1] * len(_input)

    next = 0
    limit = 15
    c = 0
    # Evaluate the model
    with torch.no_grad():
        while next != tokenizer.sep_token_id and c < limit:
            out = model(torch.tensor(_input).unsqueeze(0).to(device), torch.tensor(input_type_ids).unsqueeze(0).to(device), torch.tensor(att_mask).unsqueeze(0).to(device), image)

            preds = torch.argmax(out[0], dim=1).tolist()
            next = preds[-1]
            _input += [next]
            input_type_ids += [0]
            att_mask += [1]
            c += 1

    print(tokenizer.decode(_input))