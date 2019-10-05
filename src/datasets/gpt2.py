import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir))
sys.path.append(root_path)

from utilities.vqa.dataset import *
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel


def build_dataset(name, directory, tokenizer, tr_size=None, ts_size=None, q_len_range=None,
                  a_len_range=None, seed=555):
    """
    Builds a dataset for training and testing a model.
    Data is saved as lists using pickle, hence you'll have to convert it to tensors at the right time
    Note: This code is optimized to be fast, not fancy.
    :param name: The name of the dataset
    :param directory: The directory in which to save the dataset, relative to resources/data/
    :param tokenizer: The tokenizer to use (BERT, GPT etc)
    :param tr_size: Amount of samples for the training set
    :param ts_size: Amount of samples for the testing set
    :param a_len_range: the range in which the question length should be. Inclusive left, exclusive right [)
    :param q_len_range: the range in which the answer length should be. Inclusive left, exclusive right [)
    :param seed: random seed to replicate results
    :return: The built dataset (training + testing) and the image path directory
    """

    # Set the seed
    random.seed(seed)

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

    objects = [[qa_objects_tr, tr_candidates, vqa_helper_tr, i_path_tr],
               [qa_objects_ts, ts_candidates, vqa_helper_ts, i_path_ts]]

    print('Creating candidates..')
    for qa_objects, candidates, vqa_helper, i_path in objects:
        for qa_object in tqdm(qa_objects):
            # Parse object
            obj_id, obj_q, obj_as, _ = get_qai(qa_object, vqa_helper)

            # Embed the question
            q_embed = [tokenizer.bos_token_id] + embed(tokenizer, obj_q) + [tokenizer.sep_token_id]
            # Compute the length of the question
            q_embed_len = len(q_embed)

            # Performance booster. This really helps by avoiding multiple identical operations
            prev_answer = None
            prev_answer_emb = None

            # Every question has 10 answers
            for obj_a in obj_as:

                # Try to skip embedding if possible and use cached version
                if obj_a == prev_answer:
                    a_embed = prev_answer_emb
                else:
                    # Embed the question and answer
                    a_embed = embed(tokenizer, obj_a) + [tokenizer.eos_token_id]
                    prev_answer = obj_a
                    prev_answer_emb = a_embed

                # Compute the lengths of the answer
                a_embed_len = len(a_embed)

                # Filter out depending on question / answer lengths
                if q_len_range is not None and q_embed_len not in q_len_range:
                    continue
                if a_len_range is not None and q_embed_len not in a_len_range:
                    continue

                # Generate candidates
                if a_embed_len not in candidates:
                    candidates[a_embed_len] = [[obj_id, q_embed + a_embed]]
                else:
                    candidates[a_embed_len].append([obj_id, q_embed + a_embed])

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

    # Generate token type ids. 0 = Question, 1 = Answer
    print('Getting padding size..')
    for longest_sequence, selected in [[longest_sequence_tr, selected_tr], [longest_sequence_ts, selected_ts]]:
        for sample in tqdm(selected):
            l_seq = len(sample[1])
            # Update longest sequence
            if l_seq > longest_sequence[0]:
                longest_sequence[0] = l_seq

    # Each sample now is: ID, SEQUENCE, IMAGE_ID, TOKEN_TYPES, ATT_MASK
    # Switch to numpy to exploit better indexing

    # To numpy :)
    selected_tr = np.array(selected_tr)
    selected_ts = np.array(selected_ts)

    # Pad sequences
    print('\nPadding sequences..')
    padded_seqs = pad_sequences(selected_tr[:, 1], maxlen=longest_sequence_tr[0], padding='post',
                                value=int(tokenizer.pad_token_id))
    for sample, padded_seq in zip(selected_tr, padded_seqs):
        sample[1] = padded_seq

    padded_seqs = pad_sequences(selected_ts[:, 1], maxlen=longest_sequence_ts[0], padding='post',
                                value=int(tokenizer.pad_token_id))
    for sample, padded_seq in zip(selected_ts, padded_seqs):
        sample[1] = padded_seq

    # Dump dataset
    print('Saving training dataset to {}/tr_{}'.format(directory, name))
    with open(os.path.join(directory, 'tr_' + name), 'wb') as fd:
        pickle.dump(selected_tr, fd)

    # Dump dataset
    print('Saving testing dataset to {}/ts_{}'.format(directory, name))
    with open(os.path.join(directory, 'ts_' + name), 'wb') as fd:
        pickle.dump(selected_ts, fd)

    return selected_tr, selected_ts, i_path_tr, i_path_ts


class GPT2Dataset(Dataset):
    """
    This is a dataset specifically crafted for GPT2 models
    """

    def __init__(self, directory, name, maxlen=None, split='train'):
        try:
            with open(os.path.join(directory, name), 'rb') as fd:
                self.data = pickle.load(fd)
            # Get image path
            _, _, self.i_path = get_data_paths(data_type=split)
            self.maxlen = maxlen if maxlen is not None else len(self.data)
            print('Data loaded successfully.')
        except (OSError, IOError) as e:
            print('Unable to load data. Did you build it first?', str(e))

    def get_image(self, image_id):
        return load_image(self.i_path, image_id)

    def __getitem__(self, item):
        sample = self.data[item]

        identifier = sample[0]
        sequence = torch.tensor(sample[1]).long()

        return identifier, sequence

    def __len__(self):
        return self.maxlen


if __name__ == '__main__':
    destination = os.path.join('models', 'baseline', 'answering', 'gpt2')

    tokenizer = GPT2Tokenizer.from_pretrained()
    tokenizer.add_special_tokens(
        {'pad_token': '<pad>', 'bos_token': '<bos>', 'eos_token': '<eos>', 'sep_token': '<sep>'})

    build_dataset(directory=resources_path('models', 'baseline', 'answering', 'data'),
                  name='bert_answering',
                  tokenizer=tokenizer,
                  tr_size=100000,
                  ts_size=50000)

    tr_dataset = GPT2Dataset(directory=resources_path('models', 'baseline', 'answering', 'data'),
                             name='tr_bert_answering')
    ts_dataset = GPT2Dataset(directory=resources_path('models', 'baseline', 'answering', 'data'),
                             name='ts_bert_answering')
