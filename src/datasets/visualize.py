import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir))
sys.path.append(root_path)

from datasets import captioning, gpt2, bert, vggpt2, light
from utilities.evaluation import sanity
from utilities import paths
from models.light.model import gpt2_tokenizer as light_tokenizer
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
    light_path = paths.resources_path('models', 'light')

    __data = {
        'captioning': {
            'data': [
                captioning.CaptionDataset(location=os.path.join(baseline_path, 'captioning', 'data')).data,
                captioning.CaptionDataset(location=os.path.join(baseline_path, 'captioning', 'data'),
                                          split='testing').data
            ],
            'seq_idx': [1, None],
            'img_idx': [2, 1],
            'has_ans': [True, False],
            'sep': None,
            'pad': 0,
        },
        'bert': {
            'data': [
                bert.BertDataset(location=os.path.join(baseline_path, 'answering', 'bert', 'data')).data,
                bert.BertDataset(location=os.path.join(baseline_path, 'answering', 'bert', 'data'),
                                 split='testing').data
            ],
            'seq_idx': [1, 1],
            'img_idx': [None, None],
            'has_ans': [True, False],
            'sep': bert.bert_tokenizer.sep_token_id,
            'pad': bert.bert_tokenizer.pad_token_id,
        },
        'gpt2': {
            'data': [
                gpt2.GPT2Dataset(location=os.path.join(baseline_path, 'answering', 'gpt2', 'data')).data,
                gpt2.GPT2Dataset(location=os.path.join(baseline_path, 'answering', 'gpt2', 'data'),
                                 split='testing').data
            ],
            'seq_idx': [1, 1],
            'img_idx': [None, None],
            'has_ans': [True, False],
            'sep': gpt2.gpt2_tokenizer.sep_token_id,
            'pad': gpt2.gpt2_tokenizer.pad_token_id
        },
        'vggpt2': {
            'data': [
                vggpt2.VGGPT2Dataset(location=os.path.join(vggpt2_path, 'data')).data,
                vggpt2.VGGPT2Dataset(location=os.path.join(vggpt2_path, 'data'), split='testing').data
            ],
            'seq_idx': [1, 1],
            'img_idx': [2, 2],
            'has_ans': [True, False],
            'sep': gpt2.gpt2_tokenizer.sep_token_id,
            'pad': gpt2.gpt2_tokenizer.pad_token_id
        },
        'light': {
            'data': [
                light.LightDataset(location=os.path.join(light_path, 'data')).data,
                light.LightDataset(location=os.path.join(light_path, 'data'), split='testing').data
            ],
            'seq_idx': [1, 1],
            'img_idx': [2, 3],
            'has_ans': [True, False],
            'sep': light_tokenizer._convert_token_to_id('?'),
            'pad': light_tokenizer._convert_token_to_id('-')
        }
    }

    for model, attr in __data.items():

        max_seq_len = [None, None]
        min_seq_len = [None, None]
        avg_seq_len = [None, None]

        max_q_len = [None, None]
        min_q_len = [None, None]
        avg_q_len = [None, None]

        max_a_len = [None, None]
        min_a_len = [None, None]
        avg_a_len = [None, None]

        max_p_len = [None, None]
        min_p_len = [None, None]
        avg_p_len = [None, None]

        img_freq = [None, None]

        sep = attr['sep']
        pad = attr['pad']

        for split, (elems, seq_idx, has_ans, img_idx) in \
                enumerate(
                    zip(
                        attr['data'],
                        attr['seq_idx'],
                        attr['has_ans'],
                        attr['img_idx']
                    )
                ):

            for e in elems:
                if seq_idx is not None:
                    seq = e[seq_idx]
                    __l = len(e[seq_idx])

                    if has_ans:
                        if sep is None:
                            sep_idx = len(seq)
                        else:
                            if sep in seq:
                                sep_idx = seq.index(sep)
                            else:
                                print('Warning: not sep in {}'.format(e))
                                continue

                        is_padded = pad in seq

                        q = seq[:sep_idx] if sep is not None else (seq[:seq.index(pad)] if is_padded else seq[:sep_idx])
                        a = seq[sep_idx:seq.index(pad)] if is_padded else seq[sep_idx:]
                        p = seq[seq.index(pad):] if is_padded else ''

                        __al = len(a)
                        __ql = len(q)
                        __pl = len(p)

                        items = [
                            [max_seq_len, min_seq_len, avg_seq_len],
                            [max_q_len, min_q_len, avg_q_len],
                            [max_a_len, min_a_len, avg_a_len],
                            [max_p_len, min_p_len, avg_p_len]
                        ]

                        targets = [
                            __l, __ql, __al, __pl
                        ]

                    else:

                        is_padded = pad in seq

                        q = seq[:seq.index(pad)] if is_padded else seq
                        a = None
                        p = seq[seq.index(pad):] if is_padded else ''

                        __ql = len(q)
                        __al = None
                        __pl = len(p)

                        items = [
                            [max_seq_len, min_seq_len, avg_seq_len],
                            [max_q_len, min_q_len, avg_q_len],
                            [max_p_len, min_p_len, avg_p_len]

                        ]

                        targets = [
                            __l, __ql, __pl
                        ]

                    for (__max, __min, __avg), target in zip(items, targets):
                        if __max[split] is not None:
                            __max[split] = target if __max[split] < target else __max[split]
                        else:
                            __max[split] = target

                        if __min[split] is not None:
                            __min[split] = target if __min[split] > target else __min[split]
                        else:
                            __min[split] = target

                        if __avg[split] is None:
                            __avg[split] = target
                        else:
                            __avg[split] += target

                if img_idx is not None:
                    if img_freq[split] is None:
                        img_freq[split] = Counter()
                    img_freq[split].update([e[img_idx]])

            for __avg in [avg_seq_len, avg_q_len, avg_a_len, avg_p_len]:
                if __avg[split] is not None:
                    __avg[split] /= len(elems)

        print('*' * 150)
        print('Model = {}'.format(model))
        for metric, values in zip(
                ['Seq', 'Question', 'Answer', 'Pad'],
                [
                    [max_seq_len, min_seq_len, avg_seq_len],
                    [max_q_len, min_q_len, avg_q_len],
                    [max_a_len, min_a_len, avg_a_len],
                    [max_p_len, min_p_len, avg_p_len]
                ]
        ):
            __max, __min, __avg = values

            tr, ts = __min
            print('Min {} (train, test) = {} | {}'.format(metric, tr, ts))
            tr, ts = __max
            print('Max {} (train, test) = {} | {}'.format(metric, tr, ts))
            tr, ts = __avg
            print('Avg {} (train, test) = {} | {}'.format(metric, tr, ts))
            print('-' * 150)

        print('Img freq (train, test) = {} | {}'.format(
            (len(img_freq[0]) if img_freq[0] else None),
            (len(img_freq[1]) if img_freq[1] else None)
        ))
