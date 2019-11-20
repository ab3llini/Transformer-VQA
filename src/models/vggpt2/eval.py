import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

from models.vggpt2.model import VGGPT2
from utilities import paths
from utilities.visualization.softmap import *
from utilities.evaluation.evaluate import *
from utilities.evaluation.beam_search import *
from datasets.vggpt2 import VGGPT2Dataset
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import torch


def do_beam_search(model, question, image, beam_search_input, device, beam_size=1, maxlen=20):
    cs, rs, cs_out, rs_out = beam_search_with_softmaps(model, beam_search_input, len(gpt2_tokenizer), beam_size,
                                                       gpt2_tokenizer.eos_token_id, maxlen, device=device)

    if cs is not None:
        print('Best completed sequence:')
        print(gpt2_tokenizer.decode(cs))
        seq = torch.cat([torch.tensor(question).to(device), torch.tensor(cs).to(device)])
        return softmap_visualize(cs_out, seq, image, False)
    elif rs is not None:
        print('Best uncompleted sequence:')
        print(gpt2_tokenizer.decode(rs))
        seq = torch.cat([torch.tensor(question).to(device), torch.tensor(rs).to(device)])
        return softmap_visualize(rs_out, seq, image, False)


def eval_single_sample(model, device, dataset, index):
    beam_search_input, ground_truths, image = dataset[index]

    question = beam_search_input.args[0].tolist()
    print("Question:", gpt2_tokenizer.decode(question))
    print('Ground truths')
    for truth in ground_truths:
        print("Ground truth:", " ".join(truth))
    plt.axis('off')
    plt.imshow(image)
    plt.show()

    do_beam_search(model, question, image, beam_search_input, device)


def get_sample_image(dataset, index):
    set_seed(0)
    _, _, _, image = dataset[index]
    return image


def interactive_evaluation(question, model, device, dataset, index, beam_size=1, maxlen=20):
    set_seed(0)
    question = [gpt2_tokenizer.bos_token_id] + gpt2_tokenizer.encode(question) + [gpt2_tokenizer.sep_token_id]

    _, beam_search_input, ground_truths, image = dataset[index]

    beam_search_input.args[0] = torch.tensor(question).long()

    print("Question:", gpt2_tokenizer.decode(question))
    print('Ground truths')
    for truth in ground_truths:
        print("Ground truth:", " ".join(truth))
    plt.axis('off')
    plt.imshow(image)
    plt.show()

    return do_beam_search(model, question, image, beam_search_input, device, beam_size)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def decode_fn(pred):
    try:
        return nltk.word_tokenize(gpt2_tokenizer.decode(pred))
    except Exception as e:
        print('Exception while trying to decode {}.. Returning an empty string..'.format(pred))
        return ''


def init_model_data(checkpoint):
    # Set the current device
    device = 'cuda'

    # Which is the model basepath?
    model_basepath = resources_path('models', 'vggpt2')

    # Init models and load checkpoint. Disable training mode & move to device
    model = VGGPT2()
    model.load_state_dict(
        torch.load(os.path.join(model_basepath, 'checkpoints', checkpoint)))
    model.set_train_on(False)
    model.to(device)

    # Load testing dataset in RAM
    ts_dataset = VGGPT2Dataset(location=os.path.join(model_basepath, 'data'), split='testing', evaluating=True)

    return model, device, ts_dataset


if __name__ == '__main__':
    set_seed(0)
    model, device, dataset = init_model_data()
    set_seed(0)
    eval_single_sample(model, device, dataset, 542)
    set_seed(0)
    image, fig, words, alphas = interactive_evaluation('What is it?', model, device, dataset, 542)
    plt.imshow(alphas[0], alpha=1)
    plt.show()
