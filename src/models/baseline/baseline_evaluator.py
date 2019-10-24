import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

from datasets import captioning, gpt2, bert
from utilities.evaluation import sanity
from utilities import paths
import torch
import models.baseline.captioning.train as modelling_caption
from transformers import GPT2LMHeadModel, BertForMaskedLM
from utilities.evaluation.evaluate import compute_corpus_bleu
import seaborn as sns;

sns.set()
import matplotlib.pyplot as plt
import pandas as pd


def prepare_data(base_dir=paths.resources_path('models', 'baseline')):
    captioning_dataset_ts = captioning.CaptionDataset(
        directory=os.path.join(base_dir, 'captioning', 'data'),
        name='testing.pk',
        split='test',
        maxlen=100000
    )
    gpt2_dataset_ts = gpt2.GPT2Dataset(
        directory=os.path.join(base_dir, 'answering', 'gpt2', 'data'),
        name='testing.pk',
        split='test',
        bleu_batch=True,
        maxlen=100000

    )
    bert_dataset_ts = bert.BertDataset(
        directory=os.path.join(base_dir, 'answering', 'bert', 'data'),
        name='testing.pk',
        split='test',
        bleu_batch=True,
        maxlen=100000

    )

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

    captioning_model.load_state_dict(
        torch.load(
            os.path.join(base_dir, 'captioning', 'checkpoints', 'B_256_LR_0.0004_CHKP_EPOCH_2.pth')))

    gpt2_model.load_state_dict(
        torch.load(
            os.path.join(base_dir, 'answering', 'gpt2', 'checkpoints', 'B_64_LR_5e-05_CHKP_EPOCH_2.pth')))

    bert_model.load_state_dict(
        torch.load(
            os.path.join(base_dir, 'answering', 'bert', 'checkpoints', 'B_64_LR_5e-05_CHKP_EPOCH_2.pth')))

    print('Checkpoints loaded in RAM')

    data = {
        'captioning': {
            'dataset': captioning_dataset_ts,
            'vocab_size': len(captioning_dataset_ts.word_map),
            'stop_word': captioning_dataset_ts.word_map['<end>'],
            'model': captioning_model
        },
        'gpt2': {
            'dataset': gpt2_dataset_ts,
            'vocab_size': len(gpt2.gpt2_tokenizer),
            'stop_word': gpt2.gpt2_tokenizer.eos_token_id,
            'model': gpt2_model
        },
        'bert': {
            'dataset': bert_dataset_ts,
            'vocab_size': len(bert.bert_tokenizer),
            'stop_word': bert.bert_tokenizer.sep_token_id,
            'model': bert_model
        }
    }

    # Make sure we are evaluating across the same exact samples
    assert sanity.cross_dataset_similarity(captioning_dataset_ts, gpt2_dataset_ts, bert_dataset_ts)
    print('Cross similarity check passed: all datasets contain the same elements.')

    return data


def evaluate(data):
    results = {
        "beam_size": [],
        "model": [],
        "BLEU1": []
    }

    for model_name, parameters in data.items():
        print('Evaluating {}'.format(model_name))

        for k in [1, 2]:
            bleu, _, _ = compute_corpus_bleu(
                model=parameters['model'],
                dataset=parameters['dataset'],
                vocab_size=parameters['vocab_size'],
                beam_size=k,
                stop_word=parameters['stop_word'],
                max_len=10
            )
            results['beam_size'].append(k)
            results['model'].append(model_name)
            results['BLEU1'].append(bleu)

    results = pd.DataFrame(results)
    sns.set_style("darkgrid")
    plot = sns.lineplot(x="beam_size", dashes=False, y="BLEU1", hue="model", style="model", markers=["o"] * len(data),
                        data=results)
    plt.show()
    return plot.figure, results


if __name__ == '__main__':
    BASE_DIR = paths.resources_path('models', 'baseline')
    data = prepare_data(base_dir=BASE_DIR)
    plot, results = evaluate(data)

    # Save files
    SAVE_DIR = paths.resources_path('results', 'baseline')
    plot.savefig(os.path.join(SAVE_DIR, 'bleu1.png'))
    results.to_csv(os.path.join(SAVE_DIR, 'results_bleu1.csv'))
