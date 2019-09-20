from models.bert import model
from loaders.vqa import *
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch
from pytorch_transformers import BertConfig


# Settings
batch_size = 32
epochs = 3

# Objects
model = model.Model()
dataset = VQADataset('bert_vgg_padded_types_dataset.pk')
loader = DataLoader(dataset=dataset, pin_memory=True, shuffle=True, batch_size=32)
optimizer = Adam(model.parameters(), lr=5e-5)
loss_fn = CrossEntropyLoss(ignore_index=-1)
writer = SummaryWriter()

# Hardware
device = torch.cuda.current_device()
model.to(device)

# Initialize the model
model.zero_grad()

# Epochs loop
for _ in range(epochs):
    for batch in tqdm(loader, desc='Iterating w/ batch size = {}'.format(batch_size)):

        # Accessing batch objects and moving them to the computing device
        token_ids = batch[0].to(device)
        token_type_ids = batch[1].to(device)
        images = batch[2].to(device)

        # Computing model output
        out = model(token_ids, token_type_ids, images)
        labels = token_ids[:, ]

        loss = loss_fn(out, token_ids[1:].contiguous())