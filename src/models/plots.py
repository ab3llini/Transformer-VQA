import collections
import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir))
sys.path.append(root_path)

import numpy as np
from utilities import paths
from utilities.paths import resources_path, data_path, models_path
import seaborn as sns;
import json
import pandas as pd

sns.set()
import matplotlib.pyplot as plt


def get_models_in(directory):
    return [m.split('.')[0] for m in os.listdir(resources_path(directory)) if
            not (len(m.split('_')) > 1 and m.split('_')[0] == 'vqa' and m.split('_')[1] == 'ready')]


def visualize(source='results'):
    bleu_scores = {}
    wm_scores = {}
    length_scores = {}
    accuracies = {}
    model_names = get_models_in(source + '/bleu1')
    remapped_names = []
    remap = {
        'gpt2': 'Q+A Baseline GPT-2',
        'bert': 'Q+A Baseline BERT',
        'captioning': 'Q+I Baseline Captioning',
        'vggpt2': 'VGGPT-2',
        'resgpt2': 'ResGPT-2',
        'vqa_baseline': 'V+Q+A Baseline'
    }

    # Remap names
    for i, name in enumerate(model_names):
        if name in remap:
            remapped_names.append(remap[name])
        else:
            remapped_names.append(name)

    for bleu in [1, 2, 3, 4]:
        bleu_scores['bleu{}'.format(bleu)] = {}
        for name, remapped_name in zip(model_names, remapped_names):
            with open(paths.resources_path(source, 'bleu{}'.format(bleu), '{}.json'.format(name)), 'r') as fp:
                bleu_scores['bleu{}'.format(bleu)][remapped_name] = json.load(fp)
    for name, remapped_name in zip(model_names, remapped_names):
        with open(paths.resources_path(source, 'word_mover', '{}.json'.format(name)), 'r') as fp:
            wm_scores[remapped_name] = json.load(fp)
        with open(paths.resources_path(source, 'length', '{}.json'.format(name)), 'r') as fp:
            length_scores[remapped_name] = json.load(fp)
        with open(paths.resources_path(source, 'vqa', '{}'.format(name), 'accuracy.json'), 'r') as fp:
            accuracies[remapped_name] = json.load(fp)


    """
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
        rows = ''
        row_tmp = '#model & #0 & #1 & #2 & #3 \\'
        df = {}
        maximums = [['', 0] for n in range(4)]

        for n, (bleu_n, models) in enumerate(bleu_scores.items()):
            for model, scores in models.items():
                if model in df:
                    df[model].append(scores[fn])
                else:
                    df[model] = [scores[fn]]

                # Model row = n
                # Bleu col = int(bleu_n[-1]) - 1
                if maximums[int(bleu_n[-1]) - 1][1] < scores[fn]:
                    maximums[int(bleu_n[-1]) - 1][0] = model
                    maximums[int(bleu_n[-1]) - 1][1] = scores[fn]


        for model, bleus in df.items():
            line = row_tmp.replace('#model', model)
            for i, bleu in enumerate(bleus):
                if maximums[i][0] == model:
                    # Print in bold
                    line = line.replace(f'#{i}', '\textbf{' + '{:.3f}'.format(bleu) + '}')
                else:
                    line = line.replace(f'#{i}', '{:.3f}'.format(bleu))
            rows += repr(line) + '\n'
        template = template.replace('#ROWS', rows)
        with open(paths.resources_path(source, 'latex', 'bleu_{}_latex.tex'.format(fn)), 'w+') as fp:
            fp.write(template.replace('\'', ''))
    """

    sns.set_palette(sns.color_palette("hls", 14))

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
                if smoothing_fn in ['add-1']:
                    plot_data['Model'].append(model)
                    plot_data['Smoothing'].append(smoothing_fn)
                    plot_data['bleu{}'.format(bleu_n)].append(value)

        # plot = sns.barplot(x='Model', y='bleu{}'.format(bleu_n), hue='Smoothing', data=plot_data)
        # plot.set_title('{}'.format(bleu_n))
        # plot.figure.savefig(paths.resources_path(source, 'plots', '{}.png'.format(bleu_n)))
        # plt.show()

        bleu_plot['Model'].extend(plot_data['Model'])
        bleu_plot['Smoothing'].extend(plot_data['Smoothing'])
        bleu_plot['Value'].extend(plot_data['bleu{}'.format(bleu_n)])
        bleu_plot['Metric'].extend(['BLEU-{}'.format(bleu_n[-1])] * len(plot_data['Model']))

    # Sort
    sorted_idx = []
    for i in range(4):
        sorted_idx.append(np.argsort(-np.array(bleu_plot['Value'][i * 14:i * 14 + 14])))

    for i in range(4):
        for k, v in bleu_plot.items():
            bleu_plot[k][i * 14:i * 14 + 14] = [v[i * 14:i * 14 + 14][r] for r in sorted_idx[i]]

    dff = pd.DataFrame(bleu_plot)
    g = sns.catplot(x="Model", y="Value", col="Metric", row='Smoothing', sharey=False, data=dff,
                    kind="bar", ci=None, sharex=False)
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(90)

    g.fig.tight_layout()
    g.savefig(paths.resources_path(source, 'plots', 'bleu.png'), dpi=300)
    plt.show()
    plt.close()

    # SAVE PALETTE!!!
    palette = {k: f'C{n}' for n, k in enumerate(bleu_plot['Model'][:14])}

    def get_palette(m):
        return [palette[mm] for mm in m]


    # Visualize VM scores
    wm_counts_plot_data = {
        'Model': [],
        '% of answers': []
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
        wm_counts_plot_data['% of answers'].append(df.shape[0])
        wm_plot_data['Model'].extend([model] * df['wm'].shape[0])
        wm_plot_data['Word Mover Distance'].extend(df['wm'].tolist())

    dff = pd.DataFrame(wm_plot_data)

    g = sns.FacetGrid(dff, col="Model", hue='Model', col_wrap=4, sharey=True, sharex=True,
                      palette=get_palette(list(wm_scores.keys())))
    g.map(sns.distplot, 'Word Mover Distance', kde=False, bins=18)
    g.axes[0].set_ylabel('% of answers')
    g.axes[4].set_ylabel('% of answers')
    g.axes[8].set_ylabel('% of answers')
    g.axes[12].set_ylabel('% of answers')

    g.fig.tight_layout()
    plt.xlim(0, 10)
    g.savefig(paths.resources_path(source, 'plots', 'word_mover.png'), dpi=300)
    plt.show()
    plt.close()

    # Sort
    sorted_idx = np.argsort(-np.array(wm_counts_plot_data['% of answers']))

    for k, v in wm_counts_plot_data.items():
        wm_counts_plot_data[k] = [v[r] for r in sorted_idx]

    # Plot number of comparable WM distances
    plot = sns.barplot(x='Model', y='% of answers', data=wm_counts_plot_data,
                       palette=get_palette(list(wm_counts_plot_data['Model'])))

    plot.set_title('# answers for which we can compute WMD')
    plot.fig.tight_layout()

    for label in plot.get_xticklabels():
        label.set_rotation(90)

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
    g = sns.FacetGrid(df, col="Model", hue='Model', col_wrap=4, sharey=False, sharex=False, palette=get_palette(list(length_scores.keys())))
    g.map(sns.distplot, 'Answer length', kde=False, bins=20, hist_kws={"range": [0, 20]})
    g.axes[0].set_ylabel('Number of answers')
    g.axes[4].set_ylabel('Number of answers')
    g.axes[8].set_ylabel('Number of answers')
    g.axes[12].set_ylabel('Number of answers')

    g.fig.tight_layout()
    plt.xlim(0, 20)
    g.savefig(paths.resources_path(source, 'plots', 'lengths.png'))

    plt.show()
    plt.close()

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

    """
    # Sort
    sorted_idx = []
    for i in range(4):
        sorted_idx.append(np.argsort(-np.array(accuracy_df_common['Accuracy'][i * 14:i * 14 + 14])))

    for i in range(4):
        for k, v in accuracy_df_common.items():
            accuracy_df_common[k][i * 14:i * 14 + 14] = [v[i * 14:i * 14 + 14][r] for r in sorted_idx[i]]

    pal = get_palette(list(accuracy_df_common['Model']))
    accuracy_df_common = pd.DataFrame(accuracy_df_common)

    g = sns.catplot(x="Model", y="Accuracy", col="Type", data=accuracy_df_common, kind="bar", ci=None, sharey=False, palette=pal)

    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(90)

    for ax in g.axes.ravel():
        for p in ax.patches:
            ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                        va='center', xytext=(0, 8), textcoords='offset points', fontsize=6)

    g.fig.tight_layout()
    g.savefig(paths.resources_path(source, 'plots', 'common_accuracy.png'), dpi=300)
    plt.show()
    """

    with open(paths.resources_path(source, 'latex', 'accuracy_template.tex'), 'r') as fp:
        template = fp.read()

        rows = ''
        row_tmp = '#model & #0 & #1 & #2 & #3 \\'
        df = {}
        maximums = [['', 0] for n in range(4)]

        for model in accuracy_df_common['Model'][:14]:
            df[model] = []

        for i in range(4):
            for model, value in zip(
                    accuracy_df_common['Model'][i * 14:i * 14 + 14],
                    accuracy_df_common['Accuracy'][i * 14:i * 14 + 14]
            ):
                df[model].append(value)

                if maximums[i][1] < value:
                    maximums[i][0] = model
                    maximums[i][1] = value

        for model, accs in df.items():
            line = row_tmp.replace('#model', model)
            for i, acc in enumerate(accs):
                if maximums[i][0] == model:
                    # Print in bold
                    line = line.replace(f'#{i}', '\textbf{' + '{:.3f}'.format(acc) + '}')
                else:
                    line = line.replace(f'#{i}', '{:.3f}'.format(acc))
            rows += repr(line) + '\n'
        template = template.replace('#ROWS', rows)
        with open(paths.resources_path(source, 'latex', 'accuracy_latex.tex'), 'w+') as fp:
            fp.write(template.replace('\'', ''))


    best_k = 5
    accuracy_df_best = {
        'Model': [],
        'Question type': [],
        'Accuracy': [],
    }

    skip = ['VGG MAX+Linear+FixHead',
            'VGG AVG+Linear+FixHead',
            'Q+I Baseline Captioning',
            'Q+A Baseline BERT']

    pp = list(accuracies.keys())
    for s in skip:
        pp.remove(s)

    """
    for model, scores in accuracies.items():
        if model in skip:
            continue

        per_question_type = scores['perQuestionType']

        ordered_pqt_scored = sorted(per_question_type.items(), key=lambda kv: kv[1], reverse=True)
        ordered_pqt_scored = collections.OrderedDict(ordered_pqt_scored)

        top_k_keys = list(ordered_pqt_scored.keys())[:best_k]
        top_k_values = [float(v) / 100.0 for k, v in ordered_pqt_scored.items() if k in top_k_keys]

        accuracy_df_best['Model'].extend([model] * best_k)
        accuracy_df_best['Question type'].extend(top_k_keys)
        accuracy_df_best['Accuracy'].extend(top_k_values)

    g = sns.FacetGrid(pd.DataFrame(accuracy_df_best), col_wrap=3, col='Model', hue='Model', sharey=False,
                      height=3,
                      aspect=1.5, palette=get_palette(pp))
    g.map(sns.barplot, 'Accuracy', 'Question type')

    g.fig.tight_layout()
    g.savefig(paths.resources_path(source, 'plots', 'best_accuracy.png'), dpi=300)
    plt.show()
    plt.close()
    """
    accuracy_df_best = {
        'Model': [],
        'Question type': [],
        'Accuracy': [],
    }

    best_k = 5

    ordering = accuracies['VGGPT-2']
    per_question_type = ordering['perQuestionType']
    ordered_pqt_scored = sorted(per_question_type.items(), key=lambda kv: kv[1], reverse=True)
    ordered_pqt_scored = collections.OrderedDict(ordered_pqt_scored)
    top_k_keys = list(ordered_pqt_scored.keys())[:best_k]
    top_k_values = [float(v) / 100.0 for k, v in ordered_pqt_scored.items() if k in top_k_keys]

    accuracy_df_best['Model'].extend(['VGGPT-2'] * best_k)
    accuracy_df_best['Question type'].extend(top_k_keys)
    accuracy_df_best['Accuracy'].extend(top_k_values)

    for model, scores in accuracies.items():
        if model in skip + ['VGGPT-2']:
            continue
        top_k = [float(v) / 100.0 for k, v in scores['perQuestionType'].items() if k in top_k_keys]
        accuracy_df_best['Model'].extend([model] * best_k)
        accuracy_df_best['Question type'].extend(top_k_keys)
        accuracy_df_best['Accuracy'].extend(top_k)

    plt.figure(figsize=(10, 4))
    plot = sns.barplot(x='Question type', y='Accuracy', hue='Model', palette=get_palette(pp),
                       data=pd.DataFrame(accuracy_df_best))
    plot.set_title('VGGPT-2\'s Top-{} accuracies comparison'.format(best_k))
    plot.legend(loc='center right', bbox_to_anchor=(1.36, 0.5), ncol=1)
    plt.tight_layout()
    plot.figure.savefig(paths.resources_path(source, 'plots', 'accuracy_comparison.png'), dpi=300)
    plt.show()


if __name__ == '__main__':
    gen_plots = True
    prediction_dest = 'predictions'
    result_dest = 'results'

    visualize(
        source=result_dest
    )
