from models.baseline.vqa import model as baseline
from helpers.dataset import *
from torch.utils.tensorboard import SummaryWriter


# This file contains training procedures for our models
model = baseline.Model()
dataset = VQADataset()

num_train_epochs = 5
train_dataloader = DataLoader(dataset=dataset, batch_size=1, pin_memory=True)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
global_step = 0
tr_loss, logging_loss = 0.0, 0.0
device = 'cuda'
accumulation_steps = 1
logging_steps = 100

writer = SummaryWriter()


model.zero_grad()

for _ in range(num_train_epochs):
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        seqs = batch[0].to(device)
        images = batch[1].to(device)
        outputs = model(seqs, images, seqs)
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

torch.save(model, 'double_linear_mask_no_softmax.h5')