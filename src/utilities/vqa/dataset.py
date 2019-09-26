import os
import sys

this_folder = os.path.dirname(os.path.realpath(__file__))  # vqa
root_path = os.path.abspath(os.path.join(this_folder, os.pardir, os.pardir))
sys.path.append(root_path)

from utilities.paths import *
from utilities.vqa.vqa import *
from pytorch_transformers import BertTokenizer
import torch
from torch.utils.data import Dataset
import pickle
from torchvision import transforms
from PIL import Image
import random
import skimage.io as io
import numpy as np
from keras.preprocessing.sequence import pad_sequences


def embed(tokenizer, text):
    """
    Embeds a text sequence using BERT tokenizer
    :param text: text to be embedded
    :return: embedded sequence (text -> tokens -> ids)
    """
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))


def load_image(base_path, image_id):
    """
    Loads an image using PIL library
    :param base_path: image base path. Returned by get_data_paths
    :param image_id: image id
    :return: PIL image
    """
    return Image.open(base_path + str(image_id).zfill(12) + '.jpg')


def transform_image(image):
    """
    Resize, normalizes and transform to tensor a PIL image fot VGG compatibility
    :param image: the PIL image to transform
    :return: a tensor containing the transformed image (3x224x224)
    """
    if image.mode != 'RGB':
        raise Exception('Grayscale detection')

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transform(image)


def check_rgb(base_path, image_id):
    """
    Check whether or not the image is RGB
    :param base_path: image base path. Returned by get_data_paths
    :param image_id: image id
    :return: Boolean
    """
    return load_image(base_path, image_id).mode == 'RGB'


def get_qai(qa_object, vqa_helper):
    """
    Returns question, answer and image for a given qa object
    :param qa_object: the qa object, obtained by loadQA()
    :param vqa_helper: the helper, obtained by VQA()
    :return: a tuple containing id, question, answer and image
    """
    q_id = qa_object['question_id']
    # Access actual question rather than it's ID
    question = vqa_helper.qqa[q_id]['question']
    # Pre compute actual image path
    image_id = qa_object['image_id']
    # Extract only answers and get rid of additional annotations
    answers = [q['answer'] for q in qa_object['answers']]

    return q_id, question, answers, image_id


def get_data_paths(data_type='train'):
    """
    Returns paths to questions, answers and images
    :param data_type: Indicates which dataset to load: train = Training samples, test = Testing samples
    :return: question, answer and image paths
    """
    with open(os.path.join(this_folder, 'structure.json')) as fp:
        structure = json.load(fp)

    data_dir = data_path('vqa')

    # Loading strings
    v = structure['version']
    t = structure['task']
    o = structure['objective'][data_type]
    d = structure['data']
    p = {
        'q': structure['path']['questions'],
        'a': structure['path']['answers'],
        'i': structure['path']['images']
    }

    q = p['q'].format(data_dir, v, t, d, o)
    a = p['a'].format(data_dir, v, d, o)
    i = p['i'].format(data_dir, o, o)

    return q, a, i


def build_dataset(name, directory, tokenizer, tr_split=0.8, limit=None, q_len_range=None, a_len_range=None, seed=555):
    """
    Builds a dataset for training and testing a model.
    Data is saved as lists using pickle, hence you'll have to convert it to tensors at the right time
    Note: This code is optimized to be fast, not fancy.
    :param name: The name of the dataset
    :param directory: The directory in which to save the dataset, relative to resources/data/
    :param tokenizer: The tokenizer to use (BERT, GPT etc)
    :param tr_split: Amount of samples for the training set
    :param limit: Size of the training + validation sets combined
    :param a_len_range: the range in which the question length should be. Inclusive left, exclusive right [)
    :param q_len_range: the range in which the answer length should be. Inclusive left, exclusive right [)
    :param seed: random seed to replicate results
    :return: The built dataset (training + testing) and the image path directory
    """

    # Set the seed
    random.seed(seed)

    # Load in RAM the samples using the VQA helper tools and our path parser.
    q_path, a_path, i_path = get_data_paths()
    vqa_helper = VQA(a_path, q_path)

    # Get all the question/answer objects
    qa_objects = vqa_helper.loadQA(vqa_helper.getQuesIds())

    # Candidate objects prior to filtering
    candidates = {}  # Dictionary : <key> = answer length, <value> = id, question, answer, image
    selected = []

    print('Creating candidates..')
    for qa_object in tqdm(qa_objects):
        # Parse object
        obj_id, obj_q, obj_as, obj_i = get_qai(qa_object, vqa_helper)

        # Check RGB validity and skip if need to
        if not check_rgb(i_path, obj_i):
            continue

        # Embed the question
        q_embed = [tokenizer.cls_token_id] + embed(tokenizer, obj_q) + [tokenizer.sep_token_id]
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
                a_embed = embed(tokenizer, obj_a) + [tokenizer.sep_token_id]
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
                candidates[a_embed_len] = [[obj_id, q_embed, a_embed, obj_i]]
            else:
                candidates[a_embed_len].append([obj_id, q_embed, a_embed, obj_i])

    print('Shuffling candidates..')
    # Shuffle all the arrays in the dictionary. This is very important to balance the dataset
    for a_embed_len, _ in tqdm(candidates.items()):
        random.shuffle(candidates[a_embed_len])

    print('Selecting candidates..', 'Limit set to {}'.format(limit) if limit is not None else 'Limit deactivated')

    # Keep first the questions whose answer is longest.
    ordered_lengths = sorted(list(candidates.keys()), reverse=True)

    # Reduce progressively to limit
    for q_embed_len in ordered_lengths:
        # Add all samples in this length range. Then check limit.
        selected += candidates[q_embed_len]
        if limit is not None:
            # Check limit reach
            if len(selected) > limit:
                # Truncate if need to
                selected = selected[:limit]
                # Break operation
                break

    longest_sequence = 0

    # Generate token type ids. 0 = Question, 1 = Answer
    print('Generating token type ids..')
    for sample in tqdm(selected):
        l_q = len(sample[1])
        l_a = len(sample[2])
        sample.append([0] * l_q + [1] * l_a)  # sample[4] type_ids
        sample.append([1] * (l_q + l_a))  # sample[5] att_mask

        # Concatenate question & answer for later padding
        sample.insert(1, sample[1] + sample[2])

        # Delete question and answer single entities
        del sample[2:4]

        # Update longest sequence
        if l_q + l_a > longest_sequence:
            longest_sequence = l_q + l_a
    # Each sample now is: ID, SEQUENCE, IMAGE_ID, TOKEN_TYPES, ATT_MASK
    # Switch to numpy to exploit better indexing
    selected = np.array(selected)

    # Pad sequences
    print('\nPadding sequences..')
    padded_seqs = pad_sequences(selected[:, 1], maxlen=longest_sequence, padding='post',
                                value=int(tokenizer.pad_token_id))
    for sample, padded_seq in zip(selected, padded_seqs):
        sample[1] = padded_seq

    # Pad sequences
    print('\nPadding token types..')
    token_types = pad_sequences(selected[:, 3], maxlen=longest_sequence, padding='post',
                                value=int(1))
    for sample, types in zip(selected, token_types):
        sample[3] = types

    # Pad sequences
    print('\nPadding attention mask..')
    att_mask = pad_sequences(selected[:, 4], maxlen=longest_sequence, padding='post',
                             value=int(0))
    for sample, mask in zip(selected, att_mask):
        sample[4] = mask

    # Split in training & testing set
    dataset_size = len(selected)
    tr_limit = tr_split * float(dataset_size)
    tr_data, ts_data = selected[:int(tr_limit)], selected[int(tr_limit):]

    # Dump dataset
    print('Saving training dataset to {}tr_{}'.format(directory, name))
    with open(os.path.join(directory, 'tr_' + name), 'wb') as fd:
        pickle.dump(tr_data, fd)

    # Dump dataset
    print('Saving testing dataset to {}ts_{}'.format(directory, name))
    with open(os.path.join(directory, 'ts_' + name), 'wb') as fd:
        pickle.dump(ts_data, fd)

    return tr_data, ts_data, i_path


class BertDataset(Dataset):
    """
    This is a dataset specifically crafted for BERT models
    """
    def __init__(self, directory, name):
        try:
            with open(os.path.join(directory, name), 'rb') as fd:
                self.data = pickle.load(fd)
            # Get image path
            _, _, self.i_path = get_data_paths()
            print('Data loaded successfully.')
        except (OSError, IOError) as e:
            print('Unable to load data. Did you build it first?', str(e))

    def __getitem__(self, item):
        sample = self.data[item]

        identifier = sample[0]
        sequence = torch.tensor(sample[1]).long()
        image = transform_image(load_image(self.i_path, sample[2]))
        token_types = torch.tensor(sample[3]).long()
        att_mask = torch.tensor(sample[4]).long()

        return identifier, sequence, image, token_types, att_mask

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':

    build_dataset(directory=resources_path('models', 'bert', 'data'),
                  name='bert_1M.pk',
                  tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'),
                  tr_split=0.8,
                  limit=1000000)

    tr_dataset = BertDataset(directory=resources_path('models', 'bert', 'data'), name='tr_bert_1M.pk')
    ts_dataset = BertDataset(directory=resources_path('models', 'bert', 'data'), name='ts_bert_1M.pk')