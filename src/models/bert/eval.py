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

# Load the model
model = torch.load('checkpoints/bert_vgg_DS%_0.5_B_64_LR_5e-05_CHKP_EPOCH_2.h5')

# Load some data (seed to replicate experiments)
seed = random.seed(555)
dataset = VQADataset(fname='bert_vgg_padded_types_dataset.pk')
sample = dataset[random.randint(0, len(dataset))]

# Print the sample
print(sample)

# Hardware
device = torch.cuda.current_device()
model.to(device)

''' Evaluate the model
with torch.no_grad():
    # Accessing batch objects and moving them to the computing device
    question = sample[0].to(device)
    token_ids = sample[1].to(device)
    token_type_ids = sample[2].to(device)
    attention_mask = sample[3].to(device)
    images = sample[4].to(device)'''