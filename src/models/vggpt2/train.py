import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from utilities.training.trainer import Trainer
from utilities.paths import resources_path
from datasets.vggpt2 import VGGPT2Dataset
from modules.loss import GPT2Loss
from models.vggpt2.model import VGGPT2, gpt2_tokenizer
from utilities.vqa.dataset import *
from utilities.visualization.softmap import *
import torch

"""
def gpt2_callback_fn(output, batch, iteration, epoch, task, tb):
    softmap_fig, words = softmap_visualize(
        softmaps=output[1][0],
        sequence=batch[1][0],
        image=batch[2][0]
        show_plot=True
    )

    tb.add_figure(
        tag='IT_{}_ID_{}_{}'.format(iteration, batch[0][0].item(), task.upper()),
        figure=softmap_fig,
        global_step=epoch
    )

"""


def train(checkpoint=None):
    loss = GPT2Loss(pad_token_id=gpt2_tokenizer.pad_token_id)
    model = VGGPT2()

    model_basepath = os.path.join('models', 'vggpt2')

    if checkpoint is not None:
        model.load_state_dict(
            torch.load(resources_path(model_basepath, 'checkpoints', checkpoint)))

    model.set_train_on(True)

    tr_dataset = VGGPT2Dataset(location=resources_path(model_basepath, 'data'))

    learning_rate = 5e-5
    epochs = 20

    tb = SummaryWriter(log_dir=resources_path(model_basepath, 'runs', 'latest'))

    gpt2_trainer = Trainer(
        model=model,
        tr_dataset=tr_dataset,
        ts_dataset=None,
        optimizer=Adam(model.parameters(), lr=learning_rate),
        loss=lambda out, batch: loss(out[0], batch[0]),
        lr=learning_rate,
        batch_size=20,
        epochs=epochs,
        tensorboard=tb,
        num_workers=1,
        checkpoint_path=resources_path(model_basepath, 'checkpoints'),
        # callback_fn=lambda *args: gpt2_callback_fn(*(list(args) + [tb])),
        # callback_interval=40,
        device='cuda'
    )

    gpt2_trainer.train()


if __name__ == '__main__':
    train()
