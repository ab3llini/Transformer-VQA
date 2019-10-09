import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

from models.bert import model as bert_model
from utilities.paths import *
from utilities.training import *
from utilities.vqa.dataset import *
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from models.bert import loss as bert_loss
from transformers import BertTokenizer
from models.bert.dataset import *


def bert_logging_fn(out, batch, description):
    ret = ''
    for s in range(3):
        ret += '*' * 25 + '\n'
        ret += '{}'.format(description) + '\n'
        ret += 'Input = {}\n'.format(tokenizer.decode(batch[0][s].tolist()))
        ret += 'Output = {}\n'.format(tokenizer.decode(torch.argmax(out[s], dim=1).tolist()))
    return ret


if __name__ == '__main__':

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    decode = lambda text: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

    model = bert_model.Model()
    learning_rate = 5e-5

    bert_trainer = Trainer(
        model=model,
        tr_dataset=BertDataset(directory=resources_path('models', 'bert', 'data'), name='tr_bert_1M.pk', maxlen=64),
        ts_dataset=BertDataset(directory=resources_path('models', 'bert', 'data'), name='ts_bert_1M.pk', maxlen=64),
        optimizer=Adam(model.parameters(), lr=learning_rate),
        loss=lambda out, batch : bert_loss.loss_fn(out, batch[0]),
        lr=learning_rate,
        batch_size=64,
        batch_extractor=lambda batch: batch[1:],
        epochs=10,
        tensorboard=SummaryWriter(log_dir=resources_path('models', 'bert', 'runs')),
        checkpoint_path=resources_path('models', 'bert', 'checkpoints'),
        logging_fp=open(resources_path('models', 'bert', 'predictions', 'train.txt'), 'w+'),
        logging_fn=bert_logging_fn,
        logging_interval=10
    )

    bert_trainer.train()