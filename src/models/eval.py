import collections
import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir))
sys.path.append(root_path)

from datasets import captioning, gpt2, bert, vggpt2
from utilities.evaluation import sanity
from utilities.evaluation.evaluate_vqa import vqa_evaluation
from utilities.vqa.dataset import get_data_paths
from utilities import paths
import torch
import numpy as np
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
import gensim.downloader as api

evaluation_cache = json.load(open(paths.data_path('cache', 'evaluation.json'), 'r'))
question_path, annotation_path, _ = get_data_paths(data_type='test')


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
    bleu_path = paths.resources_path(destination, 'bleu{}'.format(bleu), '{}.json'.format(name))
    if os.path.exists(bleu_path):
        print('\t\t({}) Skipping bleu{} score.'.format(name, bleu))
    else:
        print('\t\t({}) Computing bleu{} score.'.format(name, bleu))
        score = compute_corpus_bleu(list(predictions.values()), references, bleu=bleu)
        with open(bleu_path
                , 'w+'
                  ) as fp:
            json.dump(score, fp)


def evaluate_model(name, answer_map, source, destination, wm_embeddings=None):
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

    # Word mover distances
    wm_dest = paths.resources_path(destination, 'word_mover', '{}.json'.format(name))
    if os.path.exists(wm_dest):
        print('\t\t({}) Skipping word mover distance.'.format(name))
    else:
        print('\t\t({}) Computing word mover distance.'.format(name))
        distances = compute_corpus_wm_distance(predictions, answer_map, embeddings=wm_embeddings)
        with open(wm_dest, 'w+') as fp:
            json.dump(distances, fp)

    # Lengths
    lengths_path = paths.resources_path(destination, 'length', '{}.json'.format(name))
    if os.path.exists(lengths_path):
        print('\t\t({}) Skipping lengths.'.format(name))
    else:
        print('\t\t({}) Computing lengths.'.format(name))
        lengths = compute_corpus_pred_len(predictions)
        with open(lengths_path, 'w+') as fp:
            json.dump(lengths, fp)

    for bleu, process in processes.items():
        process.join()
        print('\t\t({}) process with target bleu{} has completed'.format(name, bleu))

    print('\t\t({}) Evaluating on VQA script.'.format(name))

    vqa_path = paths.resources_path(destination, 'vqa', '{}'.format(name), 'accuracy.json')

    if os.path.exists(vqa_path):
        print('\t\t({}) Skipping accuracies.'.format(name))
    else:
        print('\t\t({}) Computing accuracies.'.format(name))
        vqa_evaluation(question_path, annotation_path,
                       paths.resources_path(source, 'beam_size_1', 'maxlen_20', 'vqa_ready_{}.json'.format(name)),
                       paths.resources_path(destination, 'vqa', '{}'.format(name)))

    print('\t\t({}) All done.'.format(name))


def evaluate(model_names, source='predictions', destination='results', wm_embeddings=None):
    processes = {}
    with open(paths.data_path('cache', 'evaluation.json'), 'r') as fp:
        answer_map = json.load(fp)
    for name in model_names:
        process = Process(target=evaluate_model, args=(name, answer_map, source, destination, wm_embeddings))
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
    accuracies = {}

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
        with open(paths.resources_path(source, 'vqa', '{}'.format(name), 'accuracy.json'), 'r') as fp:
            accuracies[name] = json.load(fp)

    """ INIT LATEX OUT """
    smooths = []
    for bleu_n, models in bleu_scores.items():
        for model, scores in models.items():
            for smoothing_fn, value in scores.items():
                if smoothing_fn not in smooths:
                    smooths.append(smoothing_fn)

    for fn in smooths:
        with open(paths.resources_path(source, 'latex', 'bleu_template.tex'), 'r') as fp:
            template = fp.read()
            template = template.replace('#SMOOTH', fn)
        for bleu_n, models in bleu_scores.items():
            for model, scores in models.items():
                # print('#{}{}'.format(model, bleu_n[-1]), '====>', '{:.3f}'.format(scores[fn]))
                template = template.replace('#{}{}'.format(model, bleu_n[-1]), '{:.3f}'.format(scores[fn]))
        with open(paths.resources_path(source, 'latex', 'bleu_{}_latex.tex'.format(fn)), 'w+') as fp:
            fp.write(template)

    """ END LATEX OUT """

    # Visualize BLEU scores
    bleu_plot = {
        'Model': [],
        'Smoothing': [],
        'Value': [],
        'Metric': []
    }
    for bleu_n, models in bleu_scores.items():
        plot_data = {
            'Model': [],
            'Smoothing': [],
            'bleu{}'.format(bleu_n): []
        }
        for model, scores in models.items():
            for smoothing_fn, value in scores.items():
                if smoothing_fn not in ['no-smoothing', 'avg']:
                    plot_data['Model'].append(model)
                    plot_data['Smoothing'].append(smoothing_fn)
                    plot_data['bleu{}'.format(bleu_n)].append(value)

        """
        plot = sns.barplot(x='Model', y='bleu{}'.format(bleu_n), hue='Smoothing', data=plot_data)
        plot.set_title('{}'.format(bleu_n))
        plot.figure.savefig(paths.resources_path(source, 'plots', '{}.png'.format(bleu_n)))
        plt.show()
        """
        bleu_plot['Model'].extend(plot_data['Model'])
        bleu_plot['Smoothing'].extend(plot_data['Smoothing'])
        bleu_plot['Value'].extend(plot_data['bleu{}'.format(bleu_n)])
        bleu_plot['Metric'].extend(['{}'.format(bleu_n)] * len(plot_data['Model']))

    dff = pd.DataFrame(bleu_plot)
    g = sns.catplot(x="Model", y="Value", col="Metric", row='Smoothing', margin_titles=True, data=dff, saturation=.5,
                    kind="bar", ci=None)
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(90)

    g.fig.tight_layout()
    g.savefig(paths.resources_path(source, 'plots', 'bleu.png'), dpi=300)
    plt.show()

    # Visualize VM scores
    wm_counts_plot_data = {
        'Model': [],
        'N_WM': []
    }

    wm_plot_data = {
        'Model': [],
        'Word Mover Distance': []
    }

    print('WM scores')
    for i, (model, scores) in enumerate(wm_scores.items()):
        print('Model: {}'.format(model))
        values = list(scores.values())
        df = pd.DataFrame(values, columns=['wm'])
        with pd.option_context('mode.use_inf_as_na', True):
            df = df.dropna(subset=['wm'], how='all')
        wm_counts_plot_data['Model'].append(model)
        wm_counts_plot_data['N_WM'].append(df.shape[0])
        wm_plot_data['Model'].extend([model] * df['wm'].shape[0])
        wm_plot_data['Word Mover Distance'].extend(df['wm'].tolist())

    dff = pd.DataFrame(wm_plot_data)

    g = sns.FacetGrid(dff, col="Model", hue='Model', col_wrap=3, sharey=True, sharex=True)
    g.map(sns.distplot, 'Word Mover Distance', kde=False, bins=18)
    g.axes[0].set_ylabel('Number of answers')
    g.axes[3].set_ylabel('Number of answers')
    g.fig.tight_layout()
    g.savefig(paths.resources_path(source, 'plots', 'word_mover.png'), dpi=300)
    plt.show()

    # Plot number of comparable WM distances

    plot = sns.barplot(x='Model', y='N_WM', data=wm_counts_plot_data, saturation=.5)
    plot.set_title('Number of comparable WM Distances')
    plot.figure.savefig(paths.resources_path(source, 'plots', 'wm_counts.png'))
    plt.show()

    df = {
        'Model': [],
        'Answer length': []
    }
    # Visualize Length scores
    print('Length scores')
    for model, scores in length_scores.items():
        values = list(scores.values())
        df['Model'].extend([model] * len(values))
        df['Answer length'].extend(values)
    df = pd.DataFrame(df)
    m = df['Answer length'].min()
    M = df['Answer length'].max()
    g = sns.FacetGrid(df, col="Model", hue='Model', col_wrap=3, sharey=True, sharex=True)
    g.map(sns.distplot, 'Answer length', kde=False, bins=1 + M - m, hist_kws={"range": [m, M]})
    g.axes[0].set_ylabel('Number of answers')
    g.axes[3].set_ylabel('Number of answers')
    g.fig.tight_layout()
    g.savefig(paths.resources_path(source, 'plots', 'lengths.png'))

    plt.show()

    accuracy_df_common = {
        'Model': [],
        'Type': [],
        'Accuracy': [],
    }

    # Accuracies
    for __type in ['overall', 'other', 'yes/no', 'number']:
        for model, scores in accuracies.items():
            if __type == 'overall':
                v = float(scores['overall']) / 100.0
            else:
                v = float(scores['perAnswerType'][__type]) / 100.0

            accuracy_df_common['Model'].append(model)
            accuracy_df_common['Type'].append(__type)
            accuracy_df_common['Accuracy'].append(v)
    accuracy_df_common = pd.DataFrame(accuracy_df_common)

    g = sns.catplot(x="Model", y="Accuracy", col="Type", data=accuracy_df_common, saturation=.5, kind="bar", ci=None)

    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(90)

    for ax in g.axes.ravel():
        for p in ax.patches:
            ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                        va='center', xytext=(0, 8), textcoords='offset points')

    g.savefig(paths.resources_path(source, 'plots', 'common_accuracy.png'), dpi=300)
    plt.show()

    best_k = 10
    accuracy_df_best = {
        'Model': [],
        'Question type': [],
        'Accuracy': [],
        'TopK': []
    }
    for i in range(2):
        for model, scores in accuracies.items():
            if model in ['bert', 'captioning']:
                continue
            per_question_type = scores['perQuestionType']

            ordered_pqt_scored = sorted(per_question_type.items(), key=lambda kv: kv[1], reverse=True)
            ordered_pqt_scored = collections.OrderedDict(ordered_pqt_scored)

            top_k_keys = list(ordered_pqt_scored.keys())[best_k * i:best_k * i + best_k]
            top_k_values = [float(v) / 100.0 for k, v in ordered_pqt_scored.items() if k in top_k_keys]

            accuracy_df_best['Model'].extend([model] * best_k)
            accuracy_df_best['TopK'].extend(['1-10' if i == 0 else '11-20'] * best_k)
            accuracy_df_best['Question type'].extend(top_k_keys)
            accuracy_df_best['Accuracy'].extend(top_k_values)
    cols = np.array(sns.color_palette())[2:]

    g = sns.FacetGrid(pd.DataFrame(accuracy_df_best), row="Model", col='TopK', hue='Model', palette=cols, sharey=False,
                      height=3,
                      aspect=1.5)
    g.map(sns.barplot, 'Accuracy', 'Question type', saturation=.5)

    g.fig.tight_layout()
    g.savefig(paths.resources_path(source, 'plots', 'best_accuracy.png'), dpi=300)
    plt.show()

    accuracy_df_best = {
        'Model': [],
        'Question type': [],
        'Accuracy': [],
    }

    best_k = 10

    ordering = accuracies['vggpt2']
    per_question_type = ordering['perQuestionType']
    ordered_pqt_scored = sorted(per_question_type.items(), key=lambda kv: kv[1], reverse=True)
    ordered_pqt_scored = collections.OrderedDict(ordered_pqt_scored)
    top_k_keys = list(ordered_pqt_scored.keys())[:best_k]
    top_k_values = [float(v) / 100.0 for k, v in ordered_pqt_scored.items() if k in top_k_keys]

    for model, scores in accuracies.items():
        if model in ['bert', 'captioning', 'vggpt2']:
            continue
        top_k = [float(v) / 100.0 for k, v in scores['perQuestionType'].items() if k in top_k_keys]
        accuracy_df_best['Model'].extend([model] * best_k)
        accuracy_df_best['Question type'].extend(top_k_keys)
        accuracy_df_best['Accuracy'].extend(top_k)

    accuracy_df_best['Model'].extend(['vggpt2'] * best_k)
    accuracy_df_best['Question type'].extend(top_k_keys)
    accuracy_df_best['Accuracy'].extend(top_k_values)

    plt.figure(figsize=(6, 10))
    cols = np.array(sns.color_palette())[2:]
    plot = sns.barplot(x='Accuracy', y='Question type', hue='Model', palette=cols,
                       data=pd.DataFrame(accuracy_df_best), saturation=.5)
    plot.set_title('VGGPT-2\'s Top-{} accuracies comparison'.format(best_k))
    plt.tight_layout()
    plot.figure.savefig(paths.resources_path(source, 'plots', 'accuracy_comparison.png'))
    plt.show()


if __name__ == '__main__':
    """
    Configuration
    """
    gen_preds = False
    gen_results = False
    gen_plots = True
    prediction_dest = 'predictions'
    result_dest = 'results'

    if gen_preds:
        generate_model_predictions(
            data=prepare_data(),
            beam_size=1,
            limit=20,
            destination=prediction_dest
        )
    if gen_results:
        print('Loading glove embeddings..')
        embs = api.load("glove-wiki-gigaword-100")
        evaluate(
            model_names=['captioning', 'bert', 'gpt2', 'vqa_baseline', 'vggpt2'],
            source=prediction_dest,
            destination=result_dest,
            wm_embeddings=embs
        )
    if gen_plots:
        visualize(
            model_names=['captioning', 'bert', 'gpt2', 'vqa_baseline', 'vggpt2'],
            source=result_dest
        )
