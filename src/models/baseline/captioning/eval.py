import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir, os.pardir))
sys.path.append(root_path)

from utilities.visualization.softmap import *
from utilities.evaluation.evaluate import *
from utilities.evaluation.beam_search import *
from models.baseline.captioning.model import CaptioningModel
from datasets.captioning import CaptionDataset
import random
import pandas as pd
from utilities import paths


def evaluate_bleu_score():
    pass


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    # Model parameters
    emb_dim = 512  # dimension of word embeddings
    attention_dim = 512  # dimension of attention linear layers
    decoder_dim = 512  # dimension of decoder RNN
    dropout = 0.5
    device = 'cuda'  # sets device for model and PyTorch tensors

    model_basepath = resources_path('models', 'baseline', 'captioning')
    word_map_file = resources_path(model_basepath, 'data', 'wordmap.2.json')

    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
        rev_word_map = {v: k for k, v in word_map.items()}

    set_seed(0)
    model = CaptioningModel(attention_dim, emb_dim, decoder_dim, word_map, dropout)
    model.load_state_dict(
        torch.load(
            os.path.join(model_basepath, 'checkpoints', 'B_100_LR_0.0004_CHKP_EPOCH_1.pth')))
    model.eval()

    ts_dataset = CaptionDataset(location=os.path.join(model_basepath, 'data'), split='testing', evaluating=True,
                                maxlen=20000)

    """
    scores, predictions, references = compute_sentences_bleu(
        model=model,
        dataset=ts_dataset,
        vocab_size=len(word_map),
        beam_size=1,
        stop_word=word_map['<end>'],
        max_len=10
    )

    for i, (score, pred, refs) in enumerate(zip(scores, predictions, references)):
        if score > 0:
            print('Object #{}, score = {}'.format(i, score))
            print([rev_word_map[w] for w in pred])
            for ref in refs:
                print([rev_word_map[w] for w in ref])
    """
    results = {
        'model': [],
        'beam_size': [],
        'BLEU1': []
    }

    bleu, _, _ = compute_corpus_bleu(
        model=model,
        dataset=ts_dataset,
        decode_fn=lambda pred: [rev_word_map[w] for w in pred],
        vocab_size=len(word_map),
        beam_size=1,
        stop_word=word_map['<end>'],
        max_len=10,
        device=device
    )

    results['beam_size'].append(1)
    results['model'].append('Captioning')
    results['BLEU1'].append(bleu)

    results = pd.DataFrame(results)

    print('Corpus bleu = {}'.format(bleu))

    SAVE_DIR = paths.resources_path('results', 'baseline', 'captioning')
    results.to_csv(os.path.join(SAVE_DIR, 'results.csv'))
