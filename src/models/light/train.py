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
from models.light.model import LightVggGpt2Avg, LightVggGpt2Max, gpt2_tokenizer
import torch


def train(data, batch_size=124):
    basepath = os.path.join('models', 'light', data['name'])

    loss = LightLoss(pad_token_id=gpt2_tokenizer._convert_token_to_id('-'))
    model = data['model']
    tr_dataset = LightDataset(resources_path(os.path.join('models', 'light', 'data')))
    ts_dataset = LightDataset(resources_path(os.path.join('models', 'light', 'data')), split='testing')
    learning_rate = 5e-4
    epochs = 200
    early_stopping = 3
    batch_size = batch_size

    trainer = Trainer(
        wandb_args={'project': 'light-vgg-models', 'name': data['name']},
        model=model,
        tr_dataset=tr_dataset,
        ts_dataset=ts_dataset,
        optimizer=Adam(model.parameters(), lr=learning_rate),
        loss=loss,
        epochs=epochs,
        early_stopping=early_stopping,
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
    available = {'vgg-gpt2-avg-fix-head': LightVggGpt2Avg(), 'vgg-gpt2-max-fix-head': LightVggGpt2Max()}

    selected = input('Which model do you want to train? ({}): '.format(
        str(['{}'.format(i) for i, _ in available.items()])))

    assert selected in available, 'Invalid selection'

    data = {'name': selected, 'model': available[selected]}

    batch_size = input('Batch size? [124]: ')
    if batch_size != '':
        train(data, batch_size=int(batch_size))
    else:
        train(data)
