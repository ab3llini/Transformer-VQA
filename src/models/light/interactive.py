import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

from utilities import paths
from torch.optim import Adam
from utilities.training.trainer import Trainer
from utilities.paths import resources_path
from datasets.light import LightDataset
from modules.loss import LightLoss
from models.light.model import LightVggGpt2, LightResGpt2, gpt2_tokenizer
import torch
from utilities import paths
from utilities.visualization.softmap import *
from utilities.evaluation.evaluate import *
from utilities.evaluation.beam_search import *

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

        vgg_path = paths.resources_path('models', 'light', 'vgg_gpt2')
        res_path = paths.resources_path('models', 'light', 'res_gpt-2')
        checkpointVGG = torch.load(os.path.join(vgg_path, 'checkpoints', 'latest', 'B_124_LR_5e-05_CHKP_EPOCH_19.pth'))
        checkpointResNet = torch.load(
            os.path.join(res_path, 'checkpoints', 'latest', 'B_100_LR_5e-05_CHKP_EPOCH_19.pth'))

        modelVGG = LightVggGpt2(inference=True)
        modelResNet = LightResGpt2(inference=True)

        modelVGG.cuda()
        modelVGG.load_state_dict(checkpointVGG)

        modelResNet.cuda()
        modelResNet.load_state_dict(checkpointResNet)

        init = True



def answer(question, image):
    global checkpointVGG
    global checkpointResNet

    init_singletons()

    # Resize and convert image to tensor
    torch.manual_seed(0)
    resized_image = resize_image(image)
    tensor_image = normalized_tensor_image(resized_image).cuda()

    # Encode question
    question_tkn = gpt2_tokenizer.encode(question)
    tensor_question = torch.tensor(question_tkn).long().cuda()
    tensor_question_len = torch.tensor(len(question_tkn)).long().cuda()

    # Predict
    ans, softmaps = None, None

    return ans, [resized_image]
