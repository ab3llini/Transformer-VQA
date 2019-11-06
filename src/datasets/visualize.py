import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir))
sys.path.append(root_path)

from datasets import captioning, gpt2, bert, vggpt2
from utilities.evaluation import sanity
from utilities import paths
import torch

import models.baseline.captioning.train as modelling_caption
import models.vggpt2.model as modelling_vggpt2

from transformers import GPT2LMHeadModel, BertForMaskedLM
from utilities.evaluation.evaluate import *
import seaborn as sns;
import nltk

sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import json
import math
from multiprocessing import Process

if __name__ == '__main__':
    baseline_path = paths.resources_path('models', 'baseline')
    vggpt2_path = paths.resources_path('models', 'vggpt2')

    datasets = {
        'captioning': captioning.CaptionDataset(location=os.path.join(baseline_path, 'captioning', 'data')),
        'bert': bert.BertDataset(location=os.path.join(baseline_path, 'answering', 'bert', 'data')),
        'gpt2': gpt2.GPT2Dataset(location=os.path.join(baseline_path, 'answering', 'gpt2', 'data')),
        'vggpt2': vggpt2.VGGPT2Dataset(location=os.path.join(vggpt2_path, 'data'))
    }

    for model, dataset in datasets.items():
        print(pd.DataFrame(dataset.data).head())