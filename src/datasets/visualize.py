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
from collections import Counter

if __name__ == '__main__':
    baseline_path = paths.resources_path('models', 'baseline')
    vggpt2_path = paths.resources_path('models', 'vggpt2')

    datasets = {
        'captioning': {
            'ds': captioning.CaptionDataset(location=os.path.join(baseline_path, 'captioning', 'data')),
            'cols': ['question_id', 'seq', 'image_path'],
            'pad': 0
        },
        'bert': {
            'ds': bert.BertDataset(location=os.path.join(baseline_path, 'answering', 'bert', 'data')),
            'cols': ['question_id', 'seq', 'token_type_ids', 'attention_mask', 'question_length'],
            'pad': bert.bert_tokenizer.pad_token_id
        },
        'gpt2': {
            'ds': gpt2.GPT2Dataset(location=os.path.join(baseline_path, 'answering', 'gpt2', 'data')),
            'cols': ['question_id', 'seq', 'question_length'],
            'pad': gpt2.gpt2_tokenizer.pad_token_id
        },
        'vggpt2': {
            'ds': vggpt2.VGGPT2Dataset(location=os.path.join(vggpt2_path, 'data')),
            'cols': ['question_id', 'seq', 'image_path', 'question_length'],
            'pad': gpt2.gpt2_tokenizer.pad_token_id
        }
    }

    for model, data in datasets.items():
        print('Model', model)
        q_freq = Counter()
        a_freq = Counter()
        pad_freq = Counter()
        seq_freq = Counter()

        seq_len = 0
        avg_q_len = 0
        max_q_len = 0
        min_q_len = 100
        avg_a_len = 0
        max_a_len = 0
        min_a_len = 100
        min_pad_len = 100
        max_pad_len = 0
        avg_pad_len = 0

        for i, entry in enumerate(data['ds'].data):
            if seq_len == 0:
                seq_len = len(entry[1])

            pad_idx = entry[1].index(data['pad']) if data['pad'] in entry[1] else None
            if pad_idx is not None:
                __len = len(entry[1][:pad_idx])
                n_pads = seq_len - __len
                if min_pad_len > n_pads:
                    min_pad_len = n_pads
                if max_pad_len < n_pads:
                    max_pad_len = n_pads
                avg_pad_len += n_pads
                pad_freq.update([n_pads])
            else:
                __len = seq_len

            if model == 'captioning':

                if max_a_len < __len:
                    max_a_len = __len
                if min_a_len > __len:
                    min_a_len = __len
                avg_a_len += __len
                a_freq.update([__len])


            else:

                q_len = entry[-1]
                a_len = __len - q_len

                q_freq.update([q_len])
                a_freq.update([a_len])
                seq_freq.update([a_len + q_len])

                if a_len < 0:
                    print('Neg!')
                if max_a_len < a_len:
                    max_a_len = a_len
                if min_a_len > a_len:
                    min_a_len = a_len

                if max_q_len < q_len:
                    max_q_len = q_len
                if min_q_len > q_len:
                    min_q_len = q_len

                avg_a_len += a_len
                avg_q_len += q_len

        avg_a_len /= len(data['ds'].data)
        avg_q_len /= len(data['ds'].data)
        avg_pad_len /= len(data['ds'].data)

        print('\tSeq len', seq_len)
        print('\t\tMin a len', min_a_len)
        print('\t\tMax a len', max_a_len)
        print('\t\t\tMin q len', min_q_len)
        print('\t\t\tMax q len', max_q_len)
        print('\t\t\t\tMin pad len', min_pad_len)
        print('\t\t\t\tMax pad len', max_pad_len)
        print('\t\tAvg a len', avg_a_len)
        print('\t\tAvg q len', avg_q_len)
        print('\t\tAvg pad len', avg_pad_len)
        print('\t\t\ta freq', a_freq)
        print('\t\t\tq freq', q_freq)
        print('\t\t\tseq freq', seq_freq)
        print('\t\t\tpad freq', pad_freq)