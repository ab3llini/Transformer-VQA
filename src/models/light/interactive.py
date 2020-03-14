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
from models.light.model import LightVggGpt2Avg, LightResGpt2
from models.light.model import gpt2_tokenizer as light_tokenizer

init = False
checkpoint, model = None, None


def init_singletons():
    global init
    if not init:
        global checkpoint
        global model

        model_path = paths.resources_path('models', 'light', 'vgg-gpt2-avg')
        # res_path = paths.resources_path('models', 'light', 'res-gpt2')
        checkpoint = torch.load(os.path.join(model_path, 'checkpoints', 'latest', 'B_100_LR_0.0005_CHKP_EPOCH_4.pth'),
                                map_location='cuda:0')

        model = LightVggGpt2Avg()
        model.cuda()
        model.load_state_dict(checkpoint)
        init = True


def predict(question, image, stop_word, max_len, device='cuda:0'):
    global model
    # Set the model in evaluation mode
    model.eval()
    model.to(device)

    with torch.no_grad():

        answer = []
        question = question.to(device)
        stop_condition = False
        its = 0

        while not stop_condition:
            out = model(question, image)
            # Get predicted words in this beam batch
            pred = torch.argmax(out[0, -1, :])

            eos = (pred.item() in stop_word)
            its += 1

            stop_condition = eos or its > max_len

            if not eos:
                # Append the predicted token to the question
                question = torch.cat([question, pred.unsqueeze(0).unsqueeze(0)], dim=1)
                # Append the predicted token to the answer
                answer.append(pred.item())

        return light_tokenizer.decode(answer)


def answer(question, image):
    init_singletons()

    # Resize and convert image to tensor
    torch.manual_seed(0)
    resized_image = resize_image(i)
    tensor_image = normalized_tensor_image(resized_image).cuda().unsqueeze(0)

    question_tkn = gpt2_tokenizer.encode(q)
    tensor_question = torch.tensor(question_tkn).long().cuda().unsqueeze(0)

    # Predict
    return predict(tensor_question, tensor_image, [light_tokenizer.eos_token_id], 20), [resized_image]


if __name__ == '__main__':
    with open(paths.resources_path('tmp', 'image.png'), 'rb') as fp:
        img = Image.open(fp)
        q = 'What do you see?'
        ans, _, = answer(q, img)
        print(ans)
