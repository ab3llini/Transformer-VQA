import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

import torch
from utilities import paths
from utilities.visualization.softmap import *
import models.staged.model as stage3

init = False

targets = {
    'Stage-3': {
        'class': stage3.StageThree,
        'checkpoint': 'model.pth'
    },
}


def init_singletons(dev='cuda:0'):
    global init
    if not init:
        global targets

        for target, data in targets.items():
            model_path = paths.resources_path('models', 'staged', 'ignite')
            checkpoint = torch.load(
                os.path.join(model_path, data['checkpoint']),
                map_location=dev)
            targets[target]['instance'] = data['class'](stage_one_checkpoint=None)
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

            all_answers[target] = {'answer': stage3.tokenizer.decode(answer)}

    return all_answers


def answer(question, image, args=None):
    init_singletons()

    # Resize and convert image to tensor
    torch.manual_seed(0)
    resized_image = resize_image(image)
    tensor_image = normalized_tensor_image(resized_image).cuda().unsqueeze(0)

    question_tkn = [stage3.tokenizer.bos_token] + \
                   stage3.tokenizer.encode(question) + \
                   [stage3.tokenizer.sep_token]
    tensor_question = torch.tensor(question_tkn).long().cuda().unsqueeze(0)

    # Predict
    out = predict(tensor_question, tensor_image, [stage3.tokenizer.eos_token_id], 20)
    for m, _ in out.items():
        out[m]['images'] = [resized_image]

    return out


if __name__ == '__main__':
    with open(paths.resources_path('tmp', 'image.png'), 'rb') as fp:
        img = Image.open(fp)
        q = 'What do you see?'
        ans, _, = answer(q, img, args=None)
        print(ans)
