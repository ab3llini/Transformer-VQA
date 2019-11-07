import sys
import os
from collections import Counter

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir, os.pardir))
sys.path.append(root_path)

from utilities.vqa.dataset import *
import keras.preprocessing as k_preproc
import random
import json
import nltk
from torch.utils.data import Dataset

"""
if __name__ == '__main__':
    base = data_path('cache')
    with open(os.path.join(base, 'testing.json'), 'r') as fp:
        test_data = json.load(fp)

    counter = {}
    uniq = 0
    for e in test_data:
        idd, _, _, _ = e
        if idd in counter:
            raise Exception('Dup found')
        else:
            counter[idd] = 1
            uniq += 1

    print(len(counter))
    print(uniq)"""


if __name__ == '__main__':
    base = resources_path('models', 'baseline', 'captioning', 'data')
    with open(os.path.join(base, 'training.json'), 'r') as fp:
        test_data = json.load(fp)

    with open(os.path.join(base, 'wordmap.old.json'), 'r') as fp:
        wordmap_bup = json.load(fp)

    with open(os.path.join(base, 'wordmap.json'), 'r') as fp:
        wordmap = json.load(fp)

    rev_wordmap = {v: k for k, v in wordmap.items()}

    for i, e in enumerate(tqdm(test_data)):
        _, seq, _ = e
        decoded = [rev_wordmap[w] for w in seq]
        encoded_backup = [wordmap_bup.get(w, wordmap_bup['<unk>']) for w in decoded]
        test_data[i][1] = encoded_backup

    with open(os.path.join(base, 'training.json'), 'w+') as fp:
        json.dump(test_data, fp)
