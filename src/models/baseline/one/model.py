import copy

import torch
import torch.nn as nn
from torchvision import models
from pytorch_transformers import GPT2LMHeadModel
from loaders.vqa import *
from preprocessors.preparation import *


# This is a vgg based encoder that uses a pre-trained vgg network
# We get rid of the last Linear layer originally trained to distribute probabilities
# over the 1000 classes of ImageNet on which it was trained.
# ----------------------------------------------------------------------------------
# For the moment the vgg network is non-trainable and acts only as a feature extractor
class ImageEncoder(nn.Module):

    def __init__(self, out_size=768, verbose=False):
        super(ImageEncoder, self).__init__()

        # Using pre-trained vgg
        self.vgg = models.vgg11(pretrained=True)

        # Depending on the model, the last layer has different input sizes
        in_size = self.vgg.classifier[6].in_features

        # Set the model in evaluation mode to deactivate the DropOut modules
        # This is IMPORTANT to have reproducible results during evaluation!
        # We are basically blocking the weight update for the network
        self.vgg.eval()

        # Make sure to disable weight update
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Reduce output space to out_size
        self.vgg.classifier[6] = nn.Linear(in_size, out_size)

        # Add a softmax
        self.vgg.classifier.add_module('softmax_mask', nn.Softmax())
        self.vgg.classifier[7].requires_grad = True

        # Put everything on cuda
        self.vgg.to('cuda')

    def forward(self, image):

        # Put the image tensor on cuda
        image = image.to('cuda')

        return self.encode(image)

    def train_info(self):
        print('VGG11 : Grad status')
        for name, param in self.vgg.named_parameters():
            if param.requires_grad:
                print('Trainable : TRUE ->', name)
            else:
                print('Trainable : FALSE ->', name)

    def encode(self, image):
        return self.vgg(image)


# This is the model. It exploits the 774M parameters GPT-2 language model.
# We multiply the layer right before the last softmax with the output of the image encoder
# The pre-trained will distribute probabilities over the whole vocab (50k+ words)
class Model(nn.Module):
    def __init__(self, out_size=768, gpt2lh=GPT2LMHeadModel.from_pretrained('gpt2')):
        super(Model, self).__init__()

        # Isolate the transformer
        self.gpt2 = copy.deepcopy(gpt2lh.transformer)

        # Isolate the large head. This will be the output layer of the model which will be fine tuned
        self.activation = copy.deepcopy(gpt2lh.lm_head)

        # We need to move the activation backend to cuda
        self.activation.to('cuda')

        # ImageEncoder
        self.image_encoder = ImageEncoder(out_size=out_size)

        # Set the model in evaluation mode to deactivate the DropOut modules
        # This is IMPORTANT to have reproducible results during evaluation!
        self.gpt2.eval()

        # Make sure to disable weight update
        for param in self.gpt2.parameters():
            param.requires_grad = False

        # Free some memory
        self.gpt2lh = None

        # Put everything on cuda
        self.gpt2.to('cuda')

    def forward(self, question, image):
        gpt2_out = self.gpt2(question)
        vgg_out = self.image_encoder(image)

        # Pointwise multiplication
        mask = torch.mul(gpt2_out[0], vgg_out)

        # Expansion
        out = self.activation(mask)

        # We will fine tune the model to distribute probabilities properly
        # return nn.Softmax(out)

        # We don't actually need to explicitly write down a softmax here
        # The softmax is integrated in the Pytorch CrossEntropyLoss function
        return out

    def grad_info(self):
        print('Model : Grad status')
        for name, param in self.named_parameters():
            if param.requires_grad:
                print('Trainable : TRUE ->', name)
                print('This layer has', param.numel(), 'weights')
            else:
                print('Trainable : FALSE ->', name)

