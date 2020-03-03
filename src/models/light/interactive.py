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
        res_path = paths.resources_path('models', 'light', '')
        checkpointVGG = torch.load(os.path.join(vgg_path, 'checkpoints', 'latest', 'B_20_LR_5e-05_CHKP_EPOCH_19.pth'))
        checkpointResNet = torch.load(os.path.join(vgg_path, 'checkpoints', 'latest', 'B_20_LR_5e-05_CHKP_EPOCH_19.pth'))

        modelVGG = LightVggGpt2()
        modelResNet = LightResGpt2()
        model.cuda().set_train_on(False)
        model.load_state_dict(checkpoint)

        init = True


def answer(question, image):
    global model

    init_singletons()

    # Resize and convert image to tensor
    torch.manual_seed(0)
    resized_image = resize_image(image)
    tensor_image = normalized_tensor_image(resized_image).cuda()

    # Encode question
    question_tkn = gpt2_tokenizer.encode(question)
    question_tkn = [gpt2_tokenizer.bos_token_id] + question_tkn + [gpt2_tokenizer.sep_token_id]
    tensor_question = torch.tensor(question_tkn).long().cuda()

    # Prepare Beam search input
    beam_input = BeamSearchInput(0, 0, tensor_question, tensor_image)

    # Predict
    ans, softmaps = do_beam_search(model, tensor_question, resized_image, beam_input)

    return ans, [resized_image, softmaps]
