import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

from torch.optim import Adam
from utilities.training.trainer import Trainer
from utilities.paths import resources_path
from datasets.light import LightDataset, pad_token
from modules.loss import LightLoss
from models.light.model import LightVggGpt2, LightResGpt2, gpt2_tokenizer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import nltk


def predict(model, dataset, decode_fn, stop_word, max_len, device='cuda:1'):
    # Set the model in evaluation mode
    model.eval()
    model.to(device)

    predictions = {}

    loader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=1,
        pin_memory=True,
        num_workers=4
    )

    with torch.no_grad():
        for it, batch in enumerate(tqdm(loader)):

            __id = batch[0]
            batch = batch[1:]
            answer = []

            batch = list(map(lambda tensor: tensor.to(device), batch))

            stop_condition = False
            its = 0

            while not stop_condition:
                out = model(*batch)
                # Get predicted words in this beam batch
                pred = torch.argmax(out[0, -1, :])

                eos = (pred.item() in stop_word)
                its += 1

                stop_condition = eos or its > max_len

                if not eos:
                    # Append the predicted token to the question
                    batch[0] = torch.cat([batch[0], pred.unsqueeze(0).unsqueeze(0)], dim=1)
                    # Append the predicted token to the answer
                    answer.append(pred.item())

            predictions[str(__id.item())] = answer

            # print('Done after {} => {}->{}'.format(its, __id, gpt2_tokenizer.decode(batch[0].squeeze(0).tolist())))
            # print('What was saved to the prediction out > {}'.format(predictions[str(__id.item())]))
            # print('After decode > {}'.format(gpt2_tokenizer.decode(predictions[str(__id.item())])))
            # print('After custom decode fn > {}'.format(decode_fn(predictions[str(__id.item())])))
    if decode_fn:
        print('Decoding & NLTK encoding predictions with the provided tokenizer..')
        predictions = dict(map(lambda item: (item[0], decode_fn(item[1])), predictions.items()))
    return predictions


def nltk_decode_light_fn(pred):
    try:
        return nltk.word_tokenize(gpt2_tokenizer.decode(pred))
    except Exception as e:
        print('Exception while trying to decode {}.. Returning an empty string..'.format(pred))
        return ''


if __name__ == '__main__':
    model = LightVggGpt2()
    model.load_state_dict(
        torch.load(resources_path(
            os.path.join('models', 'light', 'vgg-gpt2', 'checkpoints', 'latest', 'B_124_LR_5e-05_CHKP_EPOCH_19.pth'))))
    dataset = LightDataset(resources_path(os.path.join('models', 'light', 'vgg-gpt2', 'data')), split='testing',
                           evaluating=True)

    predict(model, dataset, decode_fn=nltk_decode_light_fn, max_len=20, stop_word=[gpt2_tokenizer.eos_token_id])
