import os
import sys

this_folder = os.path.dirname(os.path.realpath(__file__))  # vqa
root_path = os.path.abspath(os.path.join(this_folder, os.pardir, os.pardir))
sys.path.append(root_path)

from utilities.paths import *
from utilities.vqa.vqa import *
import pickle
from torchvision import transforms
from PIL import Image
import numpy as np


def embed(tokenizer, text):
    """
    Embeds a text sequence using BERT tokenizer
    :param text: text to be embedded
    :return: embedded sequence (text -> tokens -> ids)
    """
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))


def load_image(image_rel_path):
    """
    Loads an image using PIL library
    :param image_rel_path: Image relative path
    :return: PIL image
    """
    return Image.open(get_image_path(image_rel_path))


def get_image_path(image_rel_path):
    """
    :param image_rel_path: Image relative path
    :return: Image path
    """

    return os.path.join(data_path('vqa', 'Images'), image_rel_path)


def resize_image(image, size=224, central_fraction=1.0):
    """
    Resize, normalizes and transform to tensor a PIL image fot VGG compatibility
    :param size:
    :param central_fraction:
    :param image: the PIL image to transform
    :return: a tensor containing the transformed image (3x224x224)
    """
    if image.mode != 'RGB':
        raise Exception('Grayscale detection')

    transform = transforms.Compose([
        transforms.Resize(int(size / central_fraction)),
        transforms.CenterCrop(size)
    ])

    return transform(image)


def normalized_tensor_image(image):
    """
    Resize, normalizes and transform to tensor a PIL image fot VGG compatibility
    :param image: the PIL image to transform
    :return: a tensor containing the transformed image (3x224x224)
    """
    if image.mode != 'RGB':
        raise Exception('Grayscale detection')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transform(image)


def check_rgb(image_rel_path):
    """
    Check whether or not the image is RGB
    :param image_rel_path: rel img path
    :return: Boolean
    """
    return load_image(image_rel_path).mode == 'RGB'


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

    data_dir = data_path('vqa')

    q = p['q'].format(data_dir, v, t, d, o)
    a = p['a'].format(data_dir, v, d, o)
    i = p['i'].format(o, o)

    return q, a, i


def get_question_ids_from(dataset):
    return np.array(dataset.data)[:, 0]


def dump_selected(selected, directory, name, tr_split):
    # Split in training & testing set
    dataset_size = len(selected)
    tr_limit = tr_split * float(dataset_size)
    tr_data, ts_data = selected[:int(tr_limit)], selected[int(tr_limit):]

    # Dump dataset
    print('Saving training dataset to {}/tr_{}'.format(directory, name))
    with open(os.path.join(directory, 'tr_' + name), 'wb') as fd:
        pickle.dump(tr_data, fd)

    # Dump dataset
    print('Saving testing dataset to {}/ts_{}'.format(directory, name))
    with open(os.path.join(directory, 'ts_' + name), 'wb') as fd:
        pickle.dump(ts_data, fd)

    return tr_data, ts_data,
