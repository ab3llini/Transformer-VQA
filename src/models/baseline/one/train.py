from models.baseline.one import model as baseline
from loaders.vqa import *
import torch.optim as optim
import torch.nn as nn
from pytorch_transformers import AdamW
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter


# This file contains training procedures for our models
model = baseline.Model()
dataset = VQADataset(rebuild=True, limit=100000)

num_train_epochs = 1
train_dataloader = DataLoader(dataset=dataset, batch_size=64, pin_memory=True)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.0},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
global_step = 0
tr_loss, logging_loss = 0.0, 0.0
device = 'cuda'
accumulation_steps = 1
logging_steps = 100

writer = SummaryWriter()


model.zero_grad()

for _ in range(3):
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        seqs = batch[0].to(device)
        images = batch[1].to(device)
        outputs = model(seqs, seqs, images)
        loss = outputs[0]
        loss.backward()
        tr_loss += loss.item()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()
            global_step += 1

            if global_step % logging_steps == 0:
                print('\tLoss : {}'.format((tr_loss - logging_loss) / logging_steps))
                writer.add_scalar('loss', (tr_loss - logging_loss) / logging_steps, global_step)
                logging_loss = tr_loss

model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)