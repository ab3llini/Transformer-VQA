from collections import Counter

import torch
from torch.utils.data import Dataset
import h5py
import json
import os
from utilities.paths import *
from utilities.vqa.vqa import *
from utilities.vqa.dataset import *
import nltk


def build_dataset(name, directory, captions_per_image=5, min_word_freq=50, max_len=50, tr_size=None, ts_size=None):
    assert ts_size % captions_per_image == 0 and tr_size % captions_per_image == 0

    # Load in RAM the samples using the VQA helper tools and our path parser.
    q_path_tr, a_path_tr, i_path_tr = get_data_paths(data_type='train')
    q_path_ts, a_path_ts, i_path_ts = get_data_paths(data_type='test')

    vqa_helper_tr = VQA(a_path_tr, q_path_tr)
    vqa_helper_ts = VQA(a_path_ts, q_path_ts)

    # Get all the question/answer objects
    qa_objects_tr = vqa_helper_tr.loadQA(vqa_helper_tr.getQuesIds())
    qa_objects_ts = vqa_helper_ts.loadQA(vqa_helper_ts.getQuesIds())

    # Randomly sample tr_size and ts_size objects from both splits
    if tr_size is not None:
        qa_objects_tr = random.sample(qa_objects_tr, tr_size)
    if ts_size is not None:
        qa_objects_ts = random.sample(qa_objects_ts, ts_size)

    # For each qa_object, select caption_per_image answers
    tr_candidates, ts_candidates = {}, {}  # Dictionary : <key> = answer length, <value> = id, question, answer, image
    selected_tr, selected_ts = [], []

    # Read image paths and captions for each image
    word_freq = Counter()

    objects = [[qa_objects_tr, tr_candidates, vqa_helper_tr, i_path_tr],
               [qa_objects_ts, ts_candidates, vqa_helper_ts, i_path_ts]]

    print('Creating training and testing candidates..')
    for qa_objects, candidates, vqa_helper, i_path in objects:
        for qa_object in tqdm(qa_objects):

            # Parse object
            obj_id, obj_q, obj_as, obj_i = get_qai(qa_object, vqa_helper)

            # Check RGB validity and skip if need to
            if not check_rgb(i_path, obj_i):
                continue

            # Performance booster. This really helps by avoiding multiple identical operations
            prev_answer = None
            prev_answer_emb = None
            added_captions = 0

            # Shuffle answers (captions) before selecting them
            random.shuffle(obj_as)

            # Every question has 10 answers
            for obj_a in obj_as:

                if captions_per_image is not None and added_captions >= captions_per_image:
                    break

                # Try to skip embedding if possible and use cached version
                if obj_a == prev_answer:
                    a_tkn = prev_answer_emb
                else:
                    # Tokenize the answer
                    a_tkn = nltk.word_tokenize(obj_a)
                    prev_answer = obj_a
                    prev_answer_emb = a_tkn

                # Compute the lengths of the answer
                a_tkn_len = len(a_tkn)

                if a_tkn_len > max_len:
                    continue

                # Update word frequency
                word_freq.update(a_tkn)

                # Generate candidates
                if a_tkn_len not in candidates:
                    candidates[a_tkn_len] = [[obj_id, a_tkn, obj_i, a_tkn_len + 2]]
                else:
                    candidates[a_tkn_len].append([obj_id, a_tkn, obj_i, a_tkn_len + 2])

                added_captions += 1

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    print('Shuffling candidates..')
    # Shuffle all the arrays in the dictionary. This is very important to balance the dataset
    for candidates in [tr_candidates, ts_candidates]:
        for a_embed_len, _ in tqdm(candidates.items()):
            random.shuffle(candidates[a_embed_len])

    print('Selecting candidates..')
    # Add in order
    for candidates, selected in [[tr_candidates, selected_tr], [ts_candidates, selected_ts]]:
        # Keep first the questions whose answer is longest.
        ordered_lengths = sorted(list(candidates.keys()), reverse=True)

        for a_embed_len in ordered_lengths:
            # Add all samples in this length range.
            selected += candidates[a_embed_len]

    longest_sequence_tr, longest_sequence_ts = [0], [0]

    print('Encoding captions.. (answers, actually..)')
    for longest_sequence, selected in [[longest_sequence_tr, selected_tr], [longest_sequence_ts, selected_ts]]:
        for sample in tqdm(selected):
            # Update longest sequence
            sample[1] = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in sample[1]] + [
                word_map['<end>']]
            l_a = len(sample[1])
            if l_a > longest_sequence[0]:
                longest_sequence[0] = l_a

    # To numpy :)
    selected_tr = np.array(selected_tr)
    selected_ts = np.array(selected_ts)

    # Pad sequences
    print('\nPadding captions..')
    padded_seqs = pad_sequences(selected_tr[:, 1], maxlen=longest_sequence_tr[0], padding='post',
                                value=int(word_map['<pad>']))
    for sample, padded_seq in zip(selected_tr, padded_seqs):
        sample[1] = padded_seq

    padded_seqs = pad_sequences(selected_ts[:, 1], maxlen=longest_sequence_tr[0], padding='post',
                                value=int(word_map['<pad>']))
    for sample, padded_seq in zip(selected_ts, padded_seqs):
        sample[1] = padded_seq

    # Save word map to a JSON
    with open(os.path.join(directory, 'WORDMAP_cpi_{}_'.format(captions_per_image) + name + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Dump dataset
    print('Saving training dataset to {}/tr_{}'.format(directory, name))
    with open(os.path.join(directory, 'tr_cpi_{}_'.format(captions_per_image) + name), 'wb') as fd:
        pickle.dump(selected_tr, fd)

    # Dump dataset
    print('Saving testing dataset to {}/ts_{}'.format(directory, name))
    with open(os.path.join(directory, 'ts_cpi_{}_'.format(captions_per_image) + name), 'wb') as fd:
        pickle.dump(selected_ts, fd)

    return selected_tr, selected_ts, i_path_tr, i_path_ts


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

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
        caption = torch.tensor(sample[1]).long()
        image = transform_image(self.get_image(sample[2]), 256)
        lengths = torch.tensor([sample[3]]).long()

        if self.split == 'train':
            return identifier, caption, image, lengths
        else:
            allcaps = self.data[self.data[:, 0] == identifier]
            allcaps = np.array(allcaps[:, 1].tolist())
            allcaps = torch.tensor(allcaps).long()
            return identifier, caption, image, lengths, allcaps

    def __len__(self):
        return self.maxlen


if __name__ == '__main__':
    build_dataset(directory=resources_path('models', 'baseline', 'captioning', 'data'),
                  name='captioning',
                  captions_per_image=5,
                  max_len=50,
                  tr_size=200000,
                  ts_size=10000
                  )
