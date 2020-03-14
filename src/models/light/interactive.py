import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

import torch
from utilities import paths
from utilities.visualization.softmap import *
from utilities.evaluation.evaluate import *
from utilities.evaluation.beam_search import *
from models.light.model import *
from models.light.model import gpt2_tokenizer as light_tokenizer

init = False

targets = {
    'res-gpt2': {
        'class': LightResGpt2,
        'checkpoint': 'B_100_LR_5e-05_CHKP_EPOCH_19.pth'
    },
    'vgg-gpt2': {
        'class': LightVggGpt2,
        'checkpoint': 'B_124_LR_5e-05_CHKP_EPOCH_19.pth'
    },
    'vgg-gpt2-avg-concat': {
        'class': LightVggGpt2AvgConcat,
        'checkpoint': 'B_124_LR_0.0005_CHKP_EPOCH_4.pth'
    },
    'vgg-gpt2-avg': {
        'class': LightVggGpt2Avg,
        'checkpoint': 'B_100_LR_0.0005_CHKP_EPOCH_4.pth'
    },
    'vgg-gpt2-avg-fix-head': {
        'class': LightVggGpt2Avg,
        'checkpoint': 'B_124_LR_0.0005_CHKP_EPOCH_4.pth'
    },
    'vgg-gpt2-max': {
        'class': LightVggGpt2Max,
        'checkpoint': 'B_100_LR_0.0005_CHKP_EPOCH_4.pth'
    },
    'vgg-gpt2-max-fix-head': {
        'class': LightVggGpt2Max,
        'checkpoint': 'B_100_LR_0.0005_CHKP_EPOCH_4.pth'
    }
}


def init_singletons(dev='cuda:0'):
    global init
    if not init:
        global targets

        for target, data in targets.items():
            model_path = paths.resources_path('models', 'light', target)
            checkpoint = torch.load(
                os.path.join(model_path, 'checkpoints', 'latest', data['checkpoint']),
                map_location=dev)
            targets[target]['instance'] = data['class']()
            targets[target]['instance'].to(dev)
            targets[target]['instance'].load_state_dict(checkpoint)

        init = True


def predict(question, image, stop_word, max_len, device='cuda:0'):
    global targets

    all_answers = {}

    for target, data in targets.items():
        # Set the model in evaluation mode
        data['instance'].eval()
        data['instance'].to(device)

        with torch.no_grad():

            answer = []
            question = question.to(device)
            stop_condition = False
            its = 0

            while not stop_condition:
                out = data['instance'](question, image)
                # Get predicted words in this beam batch
                pred = torch.argmax(out[0, -1, :])

                eos = (pred.item() in stop_word)
                its += 1

                stop_condition = eos or its > max_len

                # Append the predicted token to the question
                question = torch.cat([question, pred.unsqueeze(0).unsqueeze(0)], dim=1)
                # Append the predicted token to the answer
                answer.append(pred.item())

            all_answers[target] = {'answer': light_tokenizer.decode(answer)}

    return all_answers


def answer(question, image, args=None):
    init_singletons()

    # Resize and convert image to tensor
    torch.manual_seed(0)
    resized_image = resize_image(image)
    tensor_image = normalized_tensor_image(resized_image).cuda().unsqueeze(0)

    question_tkn = gpt2_tokenizer.encode(question)
    tensor_question = torch.tensor(question_tkn).long().cuda().unsqueeze(0)

    # Predict
    out = predict(tensor_question, tensor_image, [light_tokenizer.eos_token_id], 20)
    for m, _ in out.items():
        out[m]['images'] = [resized_image]

    return out


if __name__ == '__main__':
    with open(paths.resources_path('tmp', 'image.png'), 'rb') as fp:
        img = Image.open(fp)
        q = 'What do you see?'
        ans, _, = answer(q, img, args=None)
        print(ans)
