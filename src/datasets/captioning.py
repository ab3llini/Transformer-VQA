import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir))
sys.path.append(root_path)

import torch
from utilities.vqa.dataset import *
from datasets.creator import VADatasetCreator
from torch.utils.data import Dataset
import nltk
from collections import Counter


class CaptioningDatasetCreator(VADatasetCreator):
    def __init__(self, wordmap_location, tr_size=None, ts_size=None, generation_seed=None, min_word_freq=50):
        super().__init__(tr_size, ts_size, generation_seed)
        self.wordmap_location = wordmap_location
        self.min_word_freq = min_word_freq

    def embed_fn(self, text):
        """
        Embeds a text sequence using NLTK tokenizer
        :param text: text to be embedded
        :return: embedded sequence + length
        """
        tkn = nltk.word_tokenize(text)
        return tkn, len(tkn)

    def process(self, candidates_tr, candidates_ts):

        word_freq = Counter()
        longest_tr, longest_ts = [0], [0]

        # Create embeddings
        print('Processing..')
        for candidates, longest in [[candidates_tr, longest_tr], [candidates_ts, longest_ts]]:
            for i, sample in tqdm(enumerate(candidates)):
                if longest[0] < sample[self.tkn_a_len_idx]:
                    longest[0] = sample[self.tkn_a_len_idx]
                # Compute word frequencies
                word_freq.update(sample[self.tkn_a_idx])

        # Create word map
        words = [w for w in word_freq.keys() if word_freq[w] > self.min_word_freq]
        word_map = {k: v + 1 for v, k in enumerate(words)}
        word_map['<unk>'] = len(word_map) + 1
        word_map['<start>'] = len(word_map) + 1
        word_map['<end>'] = len(word_map) + 1
        word_map['<pad>'] = 0

        # Add start, stop & unk tokens
        for candidates in [candidates_tr, candidates_ts]:
            for i, sample in tqdm(enumerate(candidates)):
                sample[self.tkn_a_idx] = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in
                                                                  sample[self.tkn_a_idx]] + [
                                             word_map['<end>']]
                # Update length
                sample[self.tkn_a_len_idx] = len(sample[self.tkn_a_idx])

        # Pad sequences
        print('Padding to size={},{}'.format(longest_tr, longest_ts))
        candidates_tr = self.pad_sequences(candidates_tr, axis=self.tkn_a_idx, value=word_map['<pad>'],
                                           maxlen=longest_tr[0])
        candidates_ts = self.pad_sequences(candidates_ts, axis=self.tkn_a_idx, value=word_map['<pad>'],
                                           maxlen=longest_ts[0])

        # Save word map to a JSON
        with open(os.path.join(self.wordmap_location, 'wordmap.json'), 'w') as j:
            json.dump(word_map, j)

        return candidates_tr, candidates_ts


class CaptionDataset(Dataset):
    def __init__(self, directory, name, maxlen=None, split='train'):
        try:
            with open(os.path.join(directory, name), 'rb') as fd:
                self.data = pickle.load(fd)
            # Get image path
            _, _, self.i_path = get_data_paths(data_type=split)
            self.maxlen = maxlen if maxlen is not None else len(self.data)
            self.split = split
            print('Data loaded successfully.')

            for e in range(len(self.data) - 1):
                assert self.data[e].shape == self.data[e + 1].shape

            print('Sanity check passed.')

        except (OSError, IOError) as e:
            print('Unable to load data. Did you build it first?', str(e))

    def get_image(self, image_id):
        return load_image(self.i_path, image_id)

    def __getitem__(self, item):

        sample = self.data[item]

        identifier = sample[0]
        image = transform_image(self.get_image(sample[1]), 256)
        caption = torch.tensor(sample[2]).long()
        length = torch.tensor([sample[3]]).long()

        return identifier, caption, image, length

    def __len__(self):
        return self.maxlen


def create():
    nltk.download('punkt')
    destination = resources_path('models', 'baseline', 'captioning', 'data')
    dsc = CaptioningDatasetCreator(destination, generation_seed=555)
    dsc.create(destination)


if __name__ == '__main__':
    create()
