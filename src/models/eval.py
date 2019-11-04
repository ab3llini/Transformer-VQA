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
    """
    assert sanity.cross_dataset_similarity(
        captioning_dataset_ts,
        gpt2_dataset_ts,
        bert_dataset_ts,
        vggpt2_dataset_ts
    )
    print('Cross similarity check passed: all datasets contain the same elements.')
    """
    return data


def gen_predictions(data, beam_size, limit):
    for model_name, parameters in data.items():
        if model_name != 'vggpt2':
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


def evaluate_bleu1(data, beams):
    results = {
        "beam_size": [],
        "model": [],
        "BLEU1": []
    }

    for model_name, parameters in data.items():
        print('Evaluating {}'.format(model_name))

        for k in beams:
            bleu, _, _ = compute_corpus_bleu(
                model=parameters['model'],
                dataset=parameters['dataset'],
                decode_fn=parameters['decode_fn'],
                vocab_size=parameters['vocab_size'],
                beam_size=k,
                stop_word=parameters['stop_word'],
                max_len=10,
            )
            results['beam_size'].append(k)
            results['model'].append(model_name)
            results['BLEU1'].append(bleu)

    return results


def gen_plot(results):
    results = pd.DataFrame(results)
    sns.set_style("darkgrid")
    plot = sns.lineplot(x="beam_size", dashes=False, y="BLEU1", hue="model", style="model", markers=["o"] * len(data),
                        data=results)
    plt.show()

    """
        plot, results = save_and_plot(evaluate_bleu1(data))

        # Save files
        SAVE_DIR = paths.resources_path('results', 'baseline')
        plot.savefig(os.path.join(SAVE_DIR, 'bleu1_beam2.png'))
        results.to_csv(os.path.join(SAVE_DIR, 'results_bleu1_beam2.csv'))
    """
    return plot.figure, results


def gen_wm_distances(data):
    with open(paths.data_path('cache', 'evaluation.json'), 'r') as fp:
        answer_map = json.load(fp)
    for model_name, parameters in data.items():
        print('Computing WM distances for {}'.format(model_name))
        with open(paths.resources_path('predictions', 'beam_size_1', 'maxlen_20', '{}.json'.format(model_name)),
                  'r') as fp:
            predictions = json.load(fp)

        distances = compute_corpus_wm_distance(predictions, answer_map)

        with open(paths.resources_path('results', 'wm_distances', 'beam_size_1', 'maxlen_20',
                                       '{}.json'.format(model_name)), 'w+') as fp:
            json.dump(distances, fp)


def gen_lengths(data):
    for model_name, parameters in data.items():
        print('Computing lengths {}'.format(model_name))
        with open(paths.resources_path('predictions', 'beam_size_1', 'maxlen_20', '{}.json'.format(model_name)),
                  'r') as fp:
            predictions = json.load(fp)

        lengths = compute_corpus_pred_len(predictions)

        with open(paths.resources_path('results', 'lengths', 'beam_size_1', 'maxlen_20',
                                       '{}.json'.format(model_name)), 'w+') as fp:
            json.dump(lengths, fp)


def plot_wm_distances():
    model_names = ['captioning', 'gpt2', 'vggpt2']
    n_models = len(model_names)
    predictions = []
    plot = {}
    for name in model_names:
        with open(paths.resources_path('results', 'wm_distances', 'beam_size_1', 'maxlen_20',
                                       '{}.json'.format(name)), 'r') as fp:
            preds = json.load(fp)
            predictions.append(list(preds.values()))

    for i, values in enumerate(zip(*predictions)):
        skip = False
        for v in values:
            if math.isinf(v):
                skip = True
                break
        if skip:
            continue
        for m, v in zip(model_names, values):
            if m in plot:
                plot[m].append(v)
            else:
                plot[m] = [v]

    df = pd.DataFrame(plot)
    print(df.head())
    print(df.describe())
    # sns.boxplot(x="model", y="wm", data=plot)
    # plt.show()


def plot_lengths():
    model_names = ['captioning', 'gpt2', 'vggpt2']
    n_models = len(model_names)
    predictions = []
    plot = {}
    for name in model_names:
        with open(paths.resources_path('results', 'lengths', 'beam_size_1', 'maxlen_20',
                                       '{}.json'.format(name)), 'r') as fp:
            preds = json.load(fp)
            predictions.append(list(preds.values()))

    for i, values in enumerate(zip(*predictions)):
        for m, v in zip(model_names, values):
            if m in plot:
                plot[m].append(v)
            else:
                plot[m] = [v]

    df = pd.DataFrame(plot)
    print(df.head())
    print(df.describe())
    # sns.boxplot(x="model", y="wm", data=plot)
    # plt.show()


if __name__ == '__main__':
    data = prepare_data()
    gen_predictions(data, beam_size=1, limit=20)
    # gen_lengths(data)
    gen_wm_distances(data)

    plot_wm_distances()
    plot_lengths()
