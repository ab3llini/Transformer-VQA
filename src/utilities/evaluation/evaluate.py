import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

from utilities.paths import *
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu
from torch.utils.data import DataLoader
from utilities.evaluation.beam_search import beam_search
from datasets.creator import MultiPurposeDataset
from tqdm import tqdm
import random
from nltk.corpus import wordnet
import nltk
from utilities.evaluation.wordnet_similarity import sentence_similarity
import gensim.downloader as api
import json

glove_embeddings = None


def compute_corpus_bleu(predictions, references, bleu=1):
    assert 0 < bleu < 5, Exception('Bleu should be in range 1-4')
    # Compute BLEU score
    smoothing = SmoothingFunction()
    scores = {}
    methods = {
        'no-smoothing': smoothing.method0,
        'add-epsilon': smoothing.method1,
        'add-1': smoothing.method2,
        'NIST-geom': smoothing.method3,
        'ChenCherry': smoothing.method4,
        'avg': smoothing.method5,
    }

    if bleu == 1:
        weights = (1, 0, 0, 0)
    elif bleu == 2:
        weights = (0.5, 0.5, 0, 0)
    elif bleu == 3:
        weights = (float(1) / float(3), float(1) / float(3), float(1) / float(3), 0)
    else:
        weights = (0.25, 0.25, 0.25, 0.25)

    for k, fn in methods.items():
        scores[k] = corpus_bleu(references, predictions, smoothing_function=fn, weights=weights)

    return scores


def generate_predictions(model, dataset: MultiPurposeDataset, decode_fn, vocab_size, beam_size, stop_word, max_len,
                         device='cuda'):
    # Set the model in evaluation mode
    model.eval()
    model.to(device)

    # Prepare references and predictions
    predictions = {}

    # Prepare sequential batch loader
    loader = DataLoader(dataset=dataset, collate_fn=dataset.collate_fn, num_workers=4,
                        shuffle=False, pin_memory=True)

    print('Beam searching with size = {}'.format(beam_size))
    for batch in tqdm(loader):
        # Make beam search
        bc, br = beam_search(model, batch[1][0], vocab_size, beam_size, stop_word, max_len, device)

        # Append best completed or best running
        predictions[batch[0][0]] = bc if bc is not None else br

    print('Decoding & NLTK encoding predictions with the provided tokenizer..')
    predictions = dict(map(lambda item: (item[0], decode_fn(item[1])), predictions.items()))

    return predictions


def compute_wm_distance(prediction, ground_truths, embeddings=None):
    if embeddings is None:
        embeddings = api.load("glove-wiki-gigaword-100")

    distance = None
    for truth in ground_truths:
        d = embeddings.wmdistance(prediction, truth)
        if distance is None:
            distance = d
        else:
            distance = d if d < distance else distance

    return distance


def compute_corpus_pred_len(predictions):
    lengths = {}
    for q_id, p in tqdm(predictions.items()):
        lengths[str(q_id)] = len(p)
    return lengths


def compute_corpus_wm_distance(predictions, answers_map, embeddings=None):
    distances = {}
    for q_id, p in tqdm(predictions.items()):
        d = compute_wm_distance(p, answers_map[str(q_id)], embeddings=embeddings)
        if str(q_id) in distances and d < distances[str(q_id)]:
            raise Exception('Duplicate key')
        else:
            distances[str(q_id)] = d
    return distances
