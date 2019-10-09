import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir, os.pardir))
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

MAP = {
    os.path.join(BASE_DIR, 'captioning'): [captioning_train, captioning_create],
    os.path.join(BASE_DIR, 'answering', 'bert'): [bert_train, bert_create],
    os.path.join(BASE_DIR, 'answering', 'gpt2'): [gpt2_train, gpt2_create]
}

FILES = ['training.pk', 'testing.pk']


def check():
    a = GPT2Dataset(directory=paths.resources_path('models', 'baseline', 'answering', 'gpt2', 'data'),
                    name='training.pk')
    b = BertDataset(directory=paths.resources_path('models', 'baseline', 'answering', 'bert', 'data'),
                    name='training.pk')
    c = CaptionDataset(directory=paths.resources_path('models', 'baseline', 'captioning', 'data'),
                       name='training.pk')

    for aa, bb, cc in zip(a.data, b.data, c.data):
        print('{} {} {}'.format(aa[0], bb[0], cc[0]))


def train():
    # Check dataset exists otherwise create
    print('Stage 1: sanity check')
    for data_path, (train_fn, create_fn) in MAP.items():
        for file in FILES:
            target = os.path.join(data_path, 'data', file)
            if not os.path.exists(target):
                print('[WARNING] File {} not found - Rebuilding dataset'.format(target))
                create_fn()
                break
            else:
                print('[OK] File {} found!'.format(target))

    print('Stage 2: sequential training')
    for data_path, (train_fn, create_fn) in MAP.items():
        print('Training {}'.format(data_path))
        train_fn()


if __name__ == '__main__':
    train()
