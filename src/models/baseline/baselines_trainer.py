import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

from models.baseline.captioning.train import train as captioning_train
from models.baseline.answering.bert.train import train as bert_train
from models.baseline.answering.gpt2.train import train as gpt2_train

from datasets.captioning import create as captioning_create, CaptionDataset
from datasets.bert import create as bert_create, BertDataset
from datasets.gpt2 import create as gpt2_create, GPT2Dataset

from utilities import paths

"""
This file acts as a script to run all the baseline trainings sequentially
"""

BASE_DIR = paths.resources_path('models', 'baseline')
TR_SIZE, TS_SIZE = 1000000, 1000000
SKIP = -1

MAP = {
    os.path.join(BASE_DIR, 'captioning'): [captioning_train, captioning_create],
    os.path.join(BASE_DIR, 'answering', 'bert'): [bert_train, bert_create],
    os.path.join(BASE_DIR, 'answering', 'gpt2'): [gpt2_train, gpt2_create]
}

FILES = ['training.pk', 'testing.pk']


def create():
    # Check dataset exists otherwise create
    print('Stage 1: sanity check')
    for data_path, (train_fn, create_fn) in MAP.items():
        sizes = [TR_SIZE, TS_SIZE]
        for i, file in enumerate(FILES):
            target = os.path.join(data_path, 'data', file)
            if not os.path.exists(target):
                print('[WARNING] File {} not found - Rebuilding it soon'.format(target))
            else:
                print('[OK] File {} found!'.format(target))
                sizes[i] = SKIP
        if sizes != [SKIP, SKIP]:
            create_fn(*sizes)


def train():
    print('Stage 2: sequential training')
    for data_path, (train_fn, create_fn) in MAP.items():
        print('Training {}'.format(data_path))
        train_fn()


if __name__ == '__main__':
    create()
    # train()
