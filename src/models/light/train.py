import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

from torch.optim import Adam
from utilities.training.trainer import Trainer
from utilities.paths import resources_path
from datasets.light import LightDataset
from modules.loss import LightLoss
from models.light.model import LightVggGpt2, gpt2_tokenizer
import torch


def train(batch_size=124):
    basepath = os.path.join('models', 'light', 'vgg-gpt2')

    loss = LightLoss(pad_token_id=gpt2_tokenizer._convert_token_to_id('-'))
    model = LightVggGpt2()
    tr_dataset = LightDataset(resources_path(basepath, 'data'))
    ts_dataset = LightDataset(resources_path(basepath, 'data'), split='testing')
    learning_rate = 5e-5
    epochs = 20
    batch_size = batch_size

    trainer = Trainer(
        wandb_args={'project': 'light-models', 'name': 'vgg-gpt2-1'},
        model=model,
        tr_dataset=tr_dataset,
        ts_dataset=ts_dataset,
        optimizer=Adam(model.parameters(), lr=learning_rate),
        loss=loss,
        epochs=epochs,
        num_workers=4,
        checkpoint_path=resources_path(basepath, 'checkpoints', 'latest'),
        device='cuda',
        shuffle=True,
        log_interval=10,
        lr=learning_rate,
        batch_size=batch_size
    )

    trainer.run()


if __name__ == '__main__':
    batch_size = input('Batch size? [124]: ')
    train(batch_size=int(batch_size)) if batch_size != '' else train()
