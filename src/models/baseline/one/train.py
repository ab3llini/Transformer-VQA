from models.baseline.one import model as baseline
from loaders.vqa import *
import torch.optim as optim
import torch.nn as nn


# This file contains training procedures for our models
model = baseline.Model()
dataset = VQADataset()

q, i = dataset[0]
q = q.to('cuda')
i = i.unsqueeze(0).to('cuda')

output = model(q, i)

print(output)