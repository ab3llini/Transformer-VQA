import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

from datasets import captioning, gpt2, bert, vgg_gpt2
from utilities.evaluation import sanity
from utilities import paths
import torch
import models.baseline.captioning.train as modelling_caption
from transformers import GPT2LMHeadModel, BertForMaskedLM
from utilities.evaluation.evaluate import compute_corpus_bleu
import seaborn as sns;
from models.vgg_gpt2.model import VGGPT2, gpt2_tokenizer

sns.set()
import matplotlib.pyplot as plt
import pandas as pd


def get_model_map():
    baseline_dir = paths.resources_path('models', 'baseline')
    vggpt2_dir = paths.resources_path('models', 'vgg_gpt2')

    captioning_dataset_ts = captioning.CaptionDataset(
        directory=os.path.join(baseline_dir, 'captioning', 'data'),
        name='testing.pk',
        split='test',
        maxlen=100000
    )
    gpt2_dataset_ts = gpt2.GPT2Dataset(
        directory=os.path.join(baseline_dir, 'answering', 'gpt2', 'data'),
        name='testing.pk',
        split='test',
        bleu_batch=True,
        maxlen=100000

    )
    bert_dataset_ts = bert.BertDataset(
        directory=os.path.join(baseline_dir, 'answering', 'bert', 'data'),
        name='testing.pk',
        split='test',
        bleu_batch=True,
        maxlen=100000

    )

    vggpt2_dataset_ts = vgg_gpt2.VGGPT2Dataset(
        directory=os.path.join(vggpt2_dir, 'data'),
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

    # Init models and load checkpoint. Disable training mode & move to device
    vggpt2_model = VGGPT2()

    captioning_model.load_state_dict(
        torch.load(
            os.path.join(baseline_dir, 'captioning', 'checkpoints', 'B_256_LR_0.0004_CHKP_EPOCH_2.pth')))

    gpt2_model.load_state_dict(
        torch.load(
            os.path.join(baseline_dir, 'answering', 'gpt2', 'checkpoints', 'B_64_LR_5e-05_CHKP_EPOCH_2.pth')))

    bert_model.load_state_dict(
        torch.load(
            os.path.join(baseline_dir, 'answering', 'bert', 'checkpoints', 'B_64_LR_5e-05_CHKP_EPOCH_2.pth')))

    vggpt2_model.load_state_dict(torch.load(os.path.join(vggpt2_dir, 'checkpoints', 'B_40_LR_5e-05_CHKP_EPOCH_7.pth')))

    # Load testing dataset in RAM

    data = {
        'ia': [
            {'captioning': {
                'dataset': captioning_dataset_ts,
                'vocab_size': len(captioning_dataset_ts.word_map),
                'stop_word': captioning_dataset_ts.word_map['<end>'],
                'model': captioning_model
            }}
        ],
        'qa': [
            {'gpt2': {
                'dataset': gpt2_dataset_ts,
                'vocab_size': len(gpt2.gpt2_tokenizer),
                'stop_word': gpt2.gpt2_tokenizer.eos_token_id,
                'model': gpt2_model
            }},
            {'bert': {
                'dataset': bert_dataset_ts,
                'vocab_size': len(bert.bert_tokenizer),
                'stop_word': bert.bert_tokenizer.sep_token_id,
                'model': bert_model
            }}
        ],
        'vqa': [
            {'vggpt2': {
                'dataset': vggpt2_dataset_ts,
                'vocab_size': len(gpt2_tokenizer),
                'stop_word': gpt2_tokenizer.eos_token_id,
                'model': vggpt2_model
            }}
        ],
    }

    return data


def execute(question_id, question, image_id, data):
    for task in data:
        if task == 'ia':
            # Captioning systems


