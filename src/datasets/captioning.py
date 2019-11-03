import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir))
sys.path.append(root_path)

import torch
from utilities.vqa.dataset import *
from datasets.creator import DatasetCreator, MultiPurposeDataset
from torch.utils.data import Dataset
import nltk
from collections import Counter
from utilities.evaluation.beam_search import BeamSearchInput


def create_datasets(base_path, min_word_freq=50):
    word_freq = Counter()
    word_map = {}

    longest_answer = {'training': 0, 'testing': 0}

    def elem_processing_fn(question_id, question, image_path, answer, split):
        answer_tkn = nltk.word_tokenize(answer)  # We pick always the first answer
        word_freq.update(answer_tkn)  # Update word frequency counts
        answer_tkn_len = len(answer_tkn)

        if longest_answer[split] < answer_tkn_len:
            longest_answer[split] = answer_tkn_len

        return [question_id, answer_tkn, image_path]

    def post_processing_fn(tr_data, ts_data):

        # Create word map
        words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
        word_map = {k: v + 1 for v, k in enumerate(words)}
        word_map['<unk>'] = len(word_map) + 1
        word_map['<start>'] = len(word_map) + 1
        word_map['<end>'] = len(word_map) + 1
        word_map['<pad>'] = 0

        print('Adding special tokens to training set..')
        for i, sample in enumerate(tqdm(tr_data)):
            sample[1] = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in
                                                 sample[1]] + [word_map['<end>']]

        print('Adding special tokens to testing set..')
        for i, sample in enumerate(tqdm(ts_data)):
            sample[1] = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in
                                                 sample[1]] + [word_map['<end>']]

        # Pad sequences
        print('Padding training sequences..')
        tr_data = DatasetCreator.pad_sequences(tr_data, axis=1, value=word_map['<pad>'],
                                               maxlen=longest_answer['training'] + 2)

        # Pad sequences
        print('Padding testing sequences..')
        ts_data = DatasetCreator.pad_sequences(ts_data, axis=1, value=word_map['<pad>'],
                                               maxlen=longest_answer['testing'] + 2)

        print('Dumping word map..')
        with open(os.path.join(base_path, 'wordmap.2.json'), 'w') as j:
            json.dump(word_map, j)

        return tr_data, ts_data

    DatasetCreator().create_together(tr_size=1000000, ts_size=200000,
                                     tr_destination=os.path.join(base_path, 'training'),
                                     ts_destination=os.path.join(base_path, 'testing'),
                                     elem_processing_fn=elem_processing_fn, post_processing_fn=post_processing_fn)


class CaptioningBeamSearchInput(BeamSearchInput):
    def get_logits(self, running_args, args):
        """
        We have to update the caplen while running the beam search
        """
        out = self.model(*running_args)
        running_args[2] = running_args[2] + 1
        args[2] = args[2] + 1
        return out[self.logits_idx], out


class CaptionDataset(MultiPurposeDataset):
    def __init__(self, location, split='training', maxlen=None, evaluating=False):
        super(CaptionDataset, self).__init__(location, split, maxlen, evaluating)

        word_map_file = resources_path(location, 'wordmap.json')
        with open(word_map_file, 'r') as j:
            self.word_map = json.load(j)

    def __getitem__(self, item):

        sample = self.data[item]
        if not self.evaluating:
            _, answer, image_path = sample
        else:
            __id, answer, image_path = sample
        image = load_image(image_rel_path=image_path)
        image = normalized_tensor_image(resize_image(image, size=256))
        length = len(answer)

        if not self.evaluating:
            # Return answer + image + length
            return torch.tensor(answer).long(), \
                   image, \
                   torch.tensor(length).long()
        else:
            start = torch.tensor([self.word_map['<start>']]).long()
            image = image
            length = torch.tensor(2).long()

            beam_input = CaptioningBeamSearchInput(0, 1, start, image, length)
            ground_truths = self.evaluation_data[str(__id)]
            return __id, beam_input, ground_truths


if __name__ == '__main__':
    path = resources_path('models', 'baseline', 'captioning', 'data')
    create_datasets(path)
