import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir, os.pardir, os.pardir))
sys.path.append(root_path)

from transformers import BertForMaskedLM
from utilities.training.trainer import *
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from models.bert import loss as bert_loss
from datasets.bert import *


def bert_logging_fn(out, batch, description):
    ret = ''
    for s in range(3):
        ret += '*' * 25 + '\n'
        ret += '{}'.format(description) + '\n'
        ret += 'Input = {}\n'.format(bert_tokenizer.decode(batch[0][s].tolist()))
        ret += 'Output = {}\n'.format(bert_tokenizer.decode(torch.argmax(out[0][s], dim=1).tolist()))
    return ret


def train():
    model_basepath = os.path.join('models', 'baseline', 'answering', 'bert')

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load pre-trained model (weights)
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    model.train()

    tr_dataset = BertDataset(directory=resources_path(model_basepath, 'data'),
                             name='training.pk')
    ts_dataset = BertDataset(directory=resources_path(model_basepath, 'data'),
                             name='testing.pk', split='test')

    learning_rate = 5e-5

    bert_trainer = Trainer(
        model=model,
        tr_dataset=tr_dataset,
        ts_dataset=ts_dataset,
        optimizer=Adam(model.parameters(), lr=learning_rate),
        loss=lambda out, batch: bert_loss.loss_fn(out[0], batch[0]),
        lr=learning_rate,
        batch_size=64,
        batch_extractor=lambda batch: batch[1:-1],  # Get rid of id & seq length
        epochs=3,
        tensorboard=SummaryWriter(log_dir=resources_path(model_basepath, 'runs')),
        checkpoint_path=resources_path(model_basepath, 'checkpoints'),
        logging_fp=None,
        logging_fn=bert_logging_fn,
        logging_interval=10
    )

    bert_trainer.train()


if __name__ == '__main__':
    train()
