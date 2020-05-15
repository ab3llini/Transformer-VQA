import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

from torch.optim import Adam
from utilities.training.trainer import Trainer
from utilities.paths import resources_path
from datasets.vggpt2v2 import VGGPT2v2Dataset
from modules.loss import VisualGPT2Loss
from models.vggpt2v2.model import VGGPTv2, gpt2_tokenizer


def train(batch_size=20):
    basepath = os.path.join('models', 'vggpt2v2')

    loss = VisualGPT2Loss(
        pad_token_id=gpt2_tokenizer._convert_token_to_id('-'),
        extract=0
    )
    model = VGGPTv2()
    tr_dataset = VGGPT2v2Dataset(resources_path(os.path.join(basepath, 'data')))
    ts_dataset = VGGPT2v2Dataset(resources_path(os.path.join(basepath, 'data')), split='testing')

    learning_rate = 5e-5
    epochs = 23
    batch_size = batch_size
    early_stopping = None
    checkpoint = resources_path(basepath, 'checkpoints', 'latest', 'VGGPTv2_bs=150_lr=5e-05_e=7.pth')

    trainer = Trainer(
        wandb_args={'project': 'light-models', 'name': 'vggpt2v2'},
        model=model,
        tr_dataset=tr_dataset,
        ts_dataset=ts_dataset,
        optimizer=Adam(model.parameters(), lr=learning_rate),
        loss=loss,
        epochs=epochs,
        early_stopping=early_stopping,
        num_workers=4,
        checkpoint_path=resources_path(basepath, 'checkpoints', 'latest'),
        load_checkpoint=checkpoint,
        device='cuda',
        shuffle=True,
        log_interval=10,
        lr=learning_rate,
        batch_size=batch_size
    )

    trainer.run()


if __name__ == '__main__':

    batch_size = input('Batch size? [20]: ')
    if batch_size != '':
        train(batch_size=int(batch_size))
    else:
        train()
