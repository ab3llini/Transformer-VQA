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
from multiprocessing import Process

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


def prepare_data(split='testing', skip=None):
    baseline_path = paths.resources_path('models', 'baseline')
    vggpt2_path = paths.resources_path('models', 'vggpt2')

    captioning_dataset_ts = captioning.CaptionDataset(location=os.path.join(baseline_path, 'captioning', 'data'),
                                                      split=split,
                                                      evaluating=True)

    gpt2_dataset_ts = gpt2.GPT2Dataset(location=os.path.join(baseline_path, 'answering', 'gpt2', 'data'),
                                       split=split,
                                       evaluating=True)

    bert_dataset_ts = bert.BertDataset(location=os.path.join(baseline_path, 'answering', 'bert', 'data'),
                                       split=split,
                                       evaluating=True)

    vggpt2_dataset_ts = vggpt2.VGGPT2Dataset(location=os.path.join(vggpt2_path, 'data'),
                                             split=split,
                                             evaluating=True)

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
            os.path.join(baseline_path, 'captioning', 'checkpoints', 'best.pth')))

    gpt2_model.load_state_dict(
        torch.load(
            os.path.join(baseline_path, 'answering', 'gpt2', 'checkpoints', 'best.pth')))

    bert_model.load_state_dict(
        torch.load(
            os.path.join(baseline_path, 'answering', 'bert', 'checkpoints', 'best.pth')))

    vggpt2_model.load_state_dict(
        torch.load(os.path.join(vggpt2_path, 'checkpoints', 'latest', 'B_20_LR_5e-05_CHKP_EPOCH_19.pth')))
    vggpt2_model.set_train_on(False)

    word_map_file = paths.resources_path(os.path.join(baseline_path, 'captioning', 'data', 'wordmap.json'))

    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
        rev_word_map = {v: k for k, v in word_map.items()}

    print('Checkpoints loaded in RAM')

    data = {
        'vggpt2': {
            'dataset': vggpt2_dataset_ts,
            'vocab_size': len(gpt2.gpt2_tokenizer),
            'decode_fn': nltk_decode_gpt2_fn,
            'stop_word': [gpt2.gpt2_tokenizer.eos_token_id, gpt2.gpt2_tokenizer.bos_token_id,
                          gpt2.gpt2_tokenizer.sep_token_id],
            'model': vggpt2_model
        },
        'gpt2': {
            'dataset': gpt2_dataset_ts,
            'vocab_size': len(gpt2.gpt2_tokenizer),
            'decode_fn': nltk_decode_gpt2_fn,
            'stop_word': [gpt2.gpt2_tokenizer.eos_token_id, gpt2.gpt2_tokenizer.bos_token_id,
                          gpt2.gpt2_tokenizer.sep_token_id],
            'model': gpt2_model
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


def generate_model_predictions(data, beam_size, limit, skip=None, destination='predictions'):
    for model_name, parameters in data.items():
        if skip is not None and model_name in skip:
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
        with open(paths.resources_path(destination,
                                       'beam_size_{}'.format(beam_size),
                                       'maxlen_{}'.format(limit),
                                       model_name + '.json'), 'w+') as fp:
            json.dump(predictions, fp)


def compute_single_bleu(bleu, name, predictions, references, destination):
    print('\t\t({}) Computing bleu{} score.'.format(name, bleu))
    score = compute_corpus_bleu(list(predictions.values()), references, bleu=bleu)
    with open(
            paths.resources_path(destination, 'bleu{}'.format(bleu), '{}.json'.format(name)), 'w+'
    ) as fp:
        json.dump(score, fp)


def evaluate_model(name, answer_map, source, destination):
    processes = {}
    print('\tEvaluating model {}'.format(name))
    with open(
            paths.resources_path(source, 'beam_size_1', 'maxlen_20', '{}.json'.format(name)),
            'r'
    ) as fp:
        predictions = json.load(fp)
        # predictions = dict((k, predictions[k]) for k in list(predictions.keys())[:500])
        references = [answer_map[p] for p in predictions]

    # Calculate bleu 1,2,3,4 with 8 different smoothing functions
    for bleu in [1, 2, 3, 4]:
        process = Process(target=compute_single_bleu, args=(bleu, name, predictions, references, destination))
        process.start()
        print('\t\t({}) process {} started with target bleu{}'.format(name, len(processes), bleu))
        processes[bleu] = process

    print('\t\t({}) Computing word mover distance.'.format(name))
    # Word mover distances

    distances = compute_corpus_wm_distance(predictions, answer_map)

    with open(paths.resources_path(destination, 'word_mover', '{}.json'.format(name)), 'w+') as fp:
        json.dump(distances, fp)

    print('\t\t({}) Computing lengths.'.format(name))
    # Lengths
    lengths = compute_corpus_pred_len(predictions)

    with open(paths.resources_path(destination, 'length', '{}.json'.format(name)), 'w+') as fp:
        json.dump(lengths, fp)

    for bleu, process in processes.items():
        process.join()
        print('\t\t({}) process with target bleu{} has completed'.format(name, bleu))

    print('\t\t({}) All done.'.format(name))


def evaluate(model_names, source='predictions', destination='results'):
    processes = {}
    with open(paths.data_path('cache', 'evaluation.json'), 'r') as fp:
        answer_map = json.load(fp)
    for name in model_names:
        process = Process(target=evaluate_model, args=(name, answer_map, source, destination))
        process.start()
        print('process {} started with target {}'.format(len(processes), name))
        processes[name] = process
    for model, process in processes.items():
        process.join()
        print('process for model {} has completed'.format(model))
    print('All done')


def visualize(model_names, source='results'):
    bleu_scores = {}
    wm_scores = {}
    length_scores = {}

    for bleu in [1, 2, 3, 4]:
        bleu_scores['bleu{}'.format(bleu)] = {}
        for name in model_names:
            with open(paths.resources_path(source, 'bleu{}'.format(bleu), '{}.json'.format(name)), 'r') as fp:
                bleu_scores['bleu{}'.format(bleu)][name] = json.load(fp)
    for name in model_names:
        with open(paths.resources_path(source, 'word_mover', '{}.json'.format(name)), 'r') as fp:
            wm_scores[name] = json.load(fp)
        with open(paths.resources_path(source, 'length', '{}.json'.format(name)), 'r') as fp:
            length_scores[name] = json.load(fp)

    # Visualize BLEU scores
    for bleu_n, models in bleu_scores.items():
        plot_data = {
            'model': [],
            'smoothing_fn': [],
            'bleu{}'.format(bleu_n): []
        }
        print('{} scores'.format(bleu_n))
        for model, scores in models.items():
            print('Model: {}'.format(model))
            for smoothing_fn, value in scores.items():
                if smoothing_fn in ['NIST-geom', 'avg']:
                    plot_data['model'].append(model)
                    plot_data['smoothing_fn'].append(smoothing_fn)
                    plot_data['bleu{}'.format(bleu_n)].append(value)
                print('Smoothing function: {} | Value = {}'.format(smoothing_fn, value))
        plot = sns.barplot(x='model', y='bleu{}'.format(bleu_n), hue='smoothing_fn', data=plot_data)
        plot.set_title('{}'.format(bleu_n))
        plot.figure.savefig(paths.resources_path(source, 'plots', '{}.png'.format(bleu_n)))
        plt.show()

    # Visualize VM scores
    wm_counts_plot_data = {
        'model': [],
        'wm_distances': []
    }
    print('WM scores')
    for model, scores in wm_scores.items():
        print('Model: {}'.format(model))
        values = list(scores.values())
        df = pd.DataFrame(values, columns=['wm'])
        with pd.option_context('mode.use_inf_as_na', True):
            df = df.dropna(subset=['wm'], how='all')
        print(df.describe())
        wm_counts_plot_data['model'].append(model)
        wm_counts_plot_data['wm_distances'].append(df.shape[0])
        plot = sns.distplot(df, kde=False)
        plot.set_title('{} - Word Mover Distance Distribution'.format(model))
        plot.set(xlabel='WM value', ylabel='Number of samples')
        plot.figure.savefig(paths.resources_path(source, 'plots', 'wm_{}.png'.format(model)))
        plt.show()

    # Plot number of comparable WM distances
    plot = sns.barplot(x='model', y='wm_distances', data=wm_counts_plot_data)
    plot.set_title('Number of comparable WM Distances')
    plot.figure.savefig(paths.resources_path(source, 'plots', 'wm_counts.png'))
    plt.show()

    df = None
    # Visualize Length scores
    print('Length scores')
    for model, scores in length_scores.items():
        print('Model: {}'.format(model))
        values = list(scores.values())
        df = pd.DataFrame(values, columns=['length'])
        print(df.describe())
        plot = sns.distplot(df, kde=False)
        plot = sns.distplot(df)
        plot.set_title('{} - Answer Length Distribution'.format(model))
        plot.figure.savefig(paths.resources_path(source, 'plots', 'length_{}.png'.format(model)))
        plt.show()


if __name__ == '__main__':
    """
    Configuration
    """
    gen_preds = True
    gen_results = True
    gen_plots = True
    prediction_dest = '100K_predictions'
    result_dest = '100K_results'

    if gen_preds:
        generate_model_predictions(
            data=prepare_data(),
            beam_size=1,
            limit=20,
            destination=prediction_dest
        )
    if gen_results:
        evaluate(
            model_names=['captioning', 'bert', 'gpt2', 'vggpt2'],
            source=prediction_dest,
            destination=result_dest
        )
    if gen_plots:
        visualize(
            model_names=['captioning', 'bert', 'gpt2', 'vggpt2'],
            source=result_dest
        )
