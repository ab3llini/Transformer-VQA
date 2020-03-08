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
from models.light.model import LightVggGpt2, LightResGpt2
from models.light.model import gpt2_tokenizer as light_tokenizer


init = False
checkpointVGG, checkpointResNet = None, None
modelVGG, modelResNet = None, None


def init_singletons():
    global init
    if not init:
        global checkpointVGG
        global checkpointResNet
        global modelVGG
        global modelResNet

        vgg_path = paths.resources_path('models', 'light', 'vgg-gpt2')
        res_path = paths.resources_path('models', 'light', 'res-gpt2')
        checkpointVGG = torch.load(os.path.join(vgg_path, 'checkpoints', 'latest', 'B_124_LR_5e-05_CHKP_EPOCH_19.pth'), map_location='cuda:0')
        checkpointResNet = torch.load(
            os.path.join(res_path, 'checkpoints', 'latest', 'B_100_LR_5e-05_CHKP_EPOCH_19.pth'),map_location='cuda:0')

        modelVGG = LightVggGpt2()
        modelResNet = LightResGpt2()

        modelVGG.cuda()
        modelVGG.load_state_dict(checkpointVGG)

        modelResNet.cuda()
        modelResNet.load_state_dict(checkpointResNet)

        init = True


def predict(model, question, image, stop_word, max_len, device='cuda:0'):
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


def a(m, q, i):
    # Resize and convert image to tensor
    torch.manual_seed(0)
    resized_image = resize_image(i)
    tensor_image = normalized_tensor_image(resized_image).cuda().unsqueeze(0)

    # Encode question
    if q is not None and len(q) > 1:
        if q[-1] == '?':
            if q[-2] != '?':
                q += '?'
        else:
            q += '??'

    print(q)

    question_tkn = gpt2_tokenizer.encode(q)
    tensor_question = torch.tensor(question_tkn).long().cuda().unsqueeze(0)

    # Predict
    return predict(m, tensor_question, tensor_image, [light_tokenizer.eos_token_id], 20), [resized_image]


def answer_vgg(question, image):
    global modelVGG
    init_singletons()
    return a(modelVGG, question, image)


def answer_res(question, image):
    global modelResNet
    init_singletons()
    return a(modelResNet, question, image)


if __name__ == '__main__':
    with open(paths.resources_path('tmp', 'image.png'), 'rb') as fp:
        img = Image.open(fp)
        q = 'What do you see?'
        ans, _, = answer_vgg(q, img)
        print(ans)

