import pandas as pd
import json
import seaborn as sns
import os
import matplotlib.pyplot as plt
import math

RESULTS_DIR = 'resources/results/word_mover/'
NAME_OVERRIDE = {'gpt2': 'Answering', 'captioning': 'Captioning', 'vqa_baseline': 'VQA', 'vggpt2': 'VGGPT-2'}



def load_wm(directory):
    try:
        results = os.listdir(directory)
        for result in results:
            with open(os.path.join(directory, result), 'r') as fp:
                values = list(json.load(fp).values())
                values = list(filter(lambda o: o != math.inf, values))
                values = list(map(lambda o: float(o), values))
                f_name, _ = os.path.splitext(result)
                fp.close()
                yield f_name, values
    except OSError as e:
        print('--> Can\'t open results!')
        raise e


def plot_compact_wm(directory):
    for f_name, values in load_wm(directory):
        if f_name in NAME_OVERRIDE:
            f_name = NAME_OVERRIDE[f_name]
            wm = pd.DataFrame(values)
            sns.distplot(wm, label=f_name, hist=False)
    plt.xlim(0, 8)
    fig = plt.gcf()
    fig.savefig('wmd.png', dpi=300)


if __name__ == '__main__':
    plt.figure(figsize=(8, 4))

    plot_compact_wm(RESULTS_DIR)
