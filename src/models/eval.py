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
from threading import Thread

evaluation_cache = json.load(open(paths.data_path('cache', 'evaluation.json'), 'r'))


def nltk_decode_gpt2_fn(pred):
    try:
        return nltk.word_tokenize(gpt2.gpt2_tokenizer.decode(pred))
    except Exception as e:
        print('Exception while trying to decode {}.. Returning an empty string..'.format(pred))
        return ''


def nltk_decode_bert_fn(pred):
    try:
        return nltk.word_tokenize(bert.bert_tokenizer.decode(pred))
    except Exception as e:
        print('Exception while trying to decode {}.. Returning an empty string..'.format(pred))
        return ''


def prepare_data(maxlen=50000, split='testing'):
    baseline_path = paths.resources_path('models', 'baseline')
    vggpt2_path = paths.resources_path('models', 'vggpt2')

    captioning_dataset_ts = captioning.CaptionDataset(location=os.path.join(baseline_path, 'captioning', 'data'),
                                                      split=split,
                                                      evaluating=True,
                                                      maxlen=maxlen
                                                      )

    gpt2_dataset_ts = gpt2.GPT2Dataset(location=os.path.join(baseline_path, 'answering', 'gpt2', 'data'),
                                       split=split,
                                       evaluating=True,
                                       maxlen=maxlen)

    bert_dataset_ts = bert.BertDataset(location=os.path.join(baseline_path, 'answering', 'bert', 'data'),
                                       split=split,
                                       evaluating=True,
                                       maxlen=maxlen)

    vggpt2_dataset_ts = vggpt2.VGGPT2Dataset(location=os.path.join(vggpt2_path, 'data'),
                                             split=split,
                                             evaluating=True,
                                             maxlen=maxlen)

    # Define model skeletons
    captioning_model = modelling_caption.CaptioningModel(
        modelling_caption.attention_dim,
        modelling_caption.emb_dim,
        modelling_caption.decoder_dim,
        captioning_dataset_ts.word_map,
        modelling_caption.dropout
    )

    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_model.resize_token_embeddings(len(gpt2.gpt2_tokenizer))
    bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    vggpt2_model = modelling_vggpt2.VGGPT2()

    captioning_model.load_state_dict(
        torch.load(
            os.path.join(baseline_path, 'captioning', 'checkpoints', 'B_100_LR_0.0004_CHKP_EPOCH_1.pth')))

    gpt2_model.load_state_dict(
        torch.load(
            os.path.join(baseline_path, 'answering', 'gpt2', 'checkpoints', 'B_64_LR_5e-05_CHKP_EPOCH_2.pth')))

    bert_model.load_state_dict(
        torch.load(
            os.path.join(baseline_path, 'answering', 'bert', 'checkpoints', 'B_64_LR_5e-05_CHKP_EPOCH_2.pth')))

    vggpt2_model.load_state_dict(
        torch.load(os.path.join(vggpt2_path, 'checkpoints', 'B_20_LR_5e-05_CHKP_EPOCH_{}.pth'.format(12))))
    vggpt2_model.set_train_on(False)

    word_map_file = paths.resources_path(os.path.join(baseline_path, 'captioning', 'data', 'wordmap.json'))

    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
        rev_word_map = {v: k for k, v in word_map.items()}

    print('Checkpoints loaded in RAM')

    data = {
        'gpt2': {
            'dataset': gpt2_dataset_ts,
            'vocab_size': len(gpt2.gpt2_tokenizer),
            'decode_fn': nltk_decode_gpt2_fn,
            'stop_word': [gpt2.gpt2_tokenizer.eos_token_id, gpt2.gpt2_tokenizer.bos_token_id,
                          gpt2.gpt2_tokenizer.sep_token_id],
            'model': gpt2_model
        },
        'vggpt2': {
            'dataset': vggpt2_dataset_ts,
            'vocab_size': len(gpt2.gpt2_tokenizer),
            'decode_fn': nltk_decode_gpt2_fn,
            'stop_word': [gpt2.gpt2_tokenizer.eos_token_id, gpt2.gpt2_tokenizer.bos_token_id,
                          gpt2.gpt2_tokenizer.sep_token_id],
            'model': vggpt2_model
        },
        'captioning': {
            'dataset': captioning_dataset_ts,
            'vocab_size': len(captioning_dataset_ts.word_map),
            'decode_fn': lambda pred: [rev_word_map[w] for w in pred],
            'stop_word': captioning_dataset_ts.word_map['<end>'],
            'model': captioning_model
        },
        'bert': {
            'dataset': bert_dataset_ts,
            'vocab_size': len(bert.bert_tokenizer),
            'decode_fn': nltk_decode_bert_fn,
            'stop_word': [bert.bert_tokenizer.cls_token_id,
                          bert.bert_tokenizer.sep_token_id],
            'model': bert_model
        }
    }

    # Make sure we are evaluating across the same exact samples

    assert sanity.cross_dataset_similarity(
        captioning_dataset_ts,
        gpt2_dataset_ts,
        bert_dataset_ts,
        vggpt2_dataset_ts
    )
    print('Cross similarity check passed: all datasets contain the same elements.')

    return data


def generate_model_predictions(data, beam_size, limit, skip=None):
    for model_name, parameters in data.items():
        if model_name in skip:
            continue
        print('Generating predictions for {}'.format(model_name))
        predictions = generate_predictions(
            model=parameters['model'],
            dataset=parameters['dataset'],
            decode_fn=parameters['decode_fn'],
            vocab_size=parameters['vocab_size'],
            beam_size=beam_size,
            stop_word=parameters['stop_word'],
            max_len=limit,
        )
        with open(paths.resources_path('predictions',
                                       'beam_size_{}'.format(beam_size),
                                       'maxlen_{}'.format(limit),
                                       model_name + '.json'), 'w+') as fp:
            json.dump(predictions, fp)


def compute_single_bleu(bleu, name, predictions, references):
    print('\t\t({}) Computing bleu{} score.'.format(name, bleu))
    score = compute_corpus_bleu(list(predictions.values()), references, bleu=bleu)
    with open(
            paths.resources_path('results', 'bleu{}'.format(bleu), '{}.json'.format(name)), 'w+'
    ) as fp:
        json.dump(score, fp)


def evaluate_model(name, answer_map):
    threads = {}
    print('\tEvaluating model {}'.format(name))
    with open(
            paths.resources_path('predictions', 'beam_size_1', 'maxlen_20', '{}.json'.format(name)),
            'r'
    ) as fp:
        predictions = json.load(fp)
        # predictions = dict((k, predictions[k]) for k in list(predictions.keys())[:500])
        references = [answer_map[p] for p in predictions]

    # Calculate bleu 1,2,3,4 with 8 different smoothing functions
    for bleu in [1, 2, 3, 4]:
        thread = Thread(target=compute_single_bleu, args=(bleu, name, predictions, references))
        thread.start()
        print('\t\t({}) Thread {} started with target bleu{}'.format(name, len(threads), bleu))
        threads[bleu] = thread

    print('\t\t({}) Computing word mover distance.'.format(name))
    # Word mover distances
    distances = compute_corpus_wm_distance(predictions, answer_map)

    with open(paths.resources_path('results', 'word_mover', '{}.json'.format(name)), 'w+') as fp:
        json.dump(distances, fp)

    print('\t\t({}) Computing lengths.'.format(name))
    # Lengths
    lengths = compute_corpus_pred_len(predictions)

    with open(paths.resources_path('results', 'length', '{}.json'.format(name)), 'w+') as fp:
        json.dump(lengths, fp)

    for bleu, thread in threads.items():
        thread.join()
        print('\t\t({}) Thread with target bleu{} has completed'.format(name, bleu))

    print('\t\t({}) All done.'.format(name))


def evaluate(model_names):
    threads = {}
    with open(paths.data_path('cache', 'evaluation.json'), 'r') as fp:
        answer_map = json.load(fp)
    for name in model_names:
        thread = Thread(target=evaluate_model, args=(name, answer_map))
        thread.start()
        print('Thread {} started with target {}'.format(len(threads), name))
        threads[name] = thread
    for model, thread in threads.items():
        thread.join()
        print('Thread for model {} has completed'.format(model))
    print('All done')


def visualize(model_names):
    bleu_scores = {}
    wm_scores = {}
    length_scores = {}

    for bleu in [1, 2, 3, 4]:
        bleu_scores['bleu{}'.format(bleu)] = {}
        for name in model_names:
            with open(paths.resources_path('results', 'bleu{}'.format(bleu), '{}.json'.format(name)), 'r') as fp:
                bleu_scores['bleu{}'.format(bleu)][name] = json.load(fp)
    for name in model_names:
        with open(paths.resources_path('results', 'word_mover', '{}.json'.format(name)), 'r') as fp:
            wm_scores[name] = json.load(fp)
        with open(paths.resources_path('results', 'length', '{}.json'.format(name)), 'r') as fp:
            length_scores[name] = json.load(fp)

    # Visualize BLEU scores
    for bleu_n, models in bleu_scores.items():
        print('{} scores'.format(bleu_n))
        for model, scores in models.items():
            print('Model: {}'.format(model))
            for smoothing_fn, value in scores.items():
                print('Smoothing function: {} | Value = {}'.format(smoothing_fn, value))

    print('WM scores')
    for model, scores in wm_scores.items():
        print('Model: {}'.format(model))
        values = list(scores.values())
        df = pd.DataFrame(values, columns=['wm'])
        with pd.option_context('mode.use_inf_as_na', True):
            df = df.dropna(subset=['wm'], how='all')
        print(df.describe())

    print('Length scores')
    for model, scores in length_scores.items():
        print('Model: {}'.format(model))
        values = list(scores.values())
        df = pd.DataFrame(values, columns=['length'])
        print(df.describe())


if __name__ == '__main__':
    # data = prepare_data()
    # generate_model_predictions(data, beam_size=1, limit=20)
    evaluate(['captioning', 'bert', 'gpt2', 'vggpt2'])
    visualize(['captioning', 'bert', 'gpt2', 'vggpt2'])
