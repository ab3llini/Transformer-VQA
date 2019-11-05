import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir, os.pardir, os.pardir))
sys.path.append(root_path)

from transformers import BertForQuestionAnswering
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
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    model.train()

    tr_dataset = BertDataset(location=resources_path(model_basepath, 'data'), split='training')

    learning_rate = 5e-5

    bert_trainer = Trainer(
        model=model,
        tr_dataset=tr_dataset,
        ts_dataset=None,
        optimizer=Adam(model.parameters(), lr=learning_rate),
        loss=lambda out, batch: bert_loss.loss_fn(out[0], batch[0]),
        lr=learning_rate,
        batch_size=20,
        epochs=10,
        num_workers=2,
        tensorboard=SummaryWriter(log_dir=resources_path(model_basepath, 'runs', 'for_QA')),
        checkpoint_path=resources_path(model_basepath, 'checkpoints', 'for_QA')
    )

    bert_trainer.train()

    del model
    del bert_tokenizer
    del bert_trainer


if __name__ == '__main__':
    train()
