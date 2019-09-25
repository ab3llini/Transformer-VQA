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
seed = random.seed(555)
dataset = VQADataset(fname='bert_vgg_padded_types_dataset.pk')
idx = random.randint(0, len(dataset))

print(dataset.samples[idx])

# Hardware
device = torch.cuda.current_device()
model.to(device)

question = torch.tensor(dataset.samples[idx].tkn_question).unsqueeze(0).to(device)
answer = dataset.samples[idx].answer
image = dataset[idx][4].unsqueeze(0).to(device)
att_mask = torch.tensor([1] * len(dataset.samples[idx].tkn_question)).unsqueeze(0).to(device)

# Print the sample


# Evaluate the model
with torch.no_grad():
    out = model(question, None, att_mask, image)
    print(tokenizer.decode(torch.argmax(out[0], dim=1).tolist()))