import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir, os.pardir, os.pardir))
sys.path.append(root_path)

from utilities import paths
import torch
from models.baseline.vqa.cyanogenoid.model import Net
import models.baseline.vqa.cyanogenoid.config as config
import models.baseline.vqa.cyanogenoid.utils as utils
from PIL import Image

from models.baseline.vqa.cyanogenoid.preprocess_images import Net as ResNet

init = False
checkpoint = None
model = None
resnet = None
answer_map = None
vocab = None


def init_singletons():
    global init
    if not init:

        global checkpoint
        global vocab
        global answer_map
        global model
        global resnet

        checkpoint = torch.load(paths.resources_path('models', 'baseline', 'vqa', 'cyanogenoid', '2017-08-04_00.55.19.pth'))

        vocab = checkpoint['vocab']

        answer_map = {v: k for k, v in vocab['answer'].items()}

        torch.manual_seed(0)

        model = Net(len(vocab['question']) + 1)
        model.to('cuda:0').eval()

        checkpoint_weight_keys = list(checkpoint['weights'].keys())
        model_weight_keys = list(model.state_dict().keys())

        for bad, good in zip(checkpoint_weight_keys, model_weight_keys):
            checkpoint['weights'][good] = checkpoint['weights'].pop(bad)

        model.load_state_dict(checkpoint['weights'])

        resnet = ResNet()
        resnet.to('cuda:0').eval()

        init = True


def answer(question, image, args=None):
    global checkpoint
    global vocab
    global answer_map
    global model
    global resnet

    init_singletons()

    # Prepare PIL image to be encoded by ResNet
    resized_image = utils.resize_image(image, config.image_size)
    tensor_image = utils.normalized_tensor_image(resized_image).unsqueeze(0).to('cuda:0')
    # Encode the image
    v = resnet(tensor_image)

    # Encode the question
    tokenized_q = question.lower()[:-1].split(' ')
    encoded_q = [vocab['question'].get(token, 0) for token in tokenized_q]
    q = torch.tensor(encoded_q).long().unsqueeze(0).to('cuda:0')
    q_len = torch.tensor(len(tokenized_q)).long().unsqueeze(0).to('cuda:0')

    # Get probabilities
    proba = model(v, q, q_len).squeeze(0)
    a = torch.argmax(proba, dim=0).item()

    # Return best answer
    return answer_map[a], [resized_image]


if __name__ == '__main__':
    init_singletons()
    with open(paths.resources_path('tmp', 'image.png'), 'rb') as fp:
        img = Image.open(fp)
        q = 'What animal is it?'
        ans = answer(q, img)
        print(ans)
