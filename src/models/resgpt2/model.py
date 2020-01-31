from torch import nn
import copy
from modules.image_encoders import ResNetEncoder101
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from modules.attention import LightAttention
import torch

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_tokenizer.add_special_tokens(
    {'pad_token': '<pad>', 'bos_token': '<bos>', 'eos_token': '<eos>', 'sep_token': '<sep>'})


class ResGPT2(nn.Module):
    def __init__(self):
        super(ResGPT2, self).__init__()

        self.map_dim = 2048  # Number of ResNet channels
        self.hidden_dim = 768  # Number of hidden parameters in GPT2 (small)

        # Resize the language model embedding layer

        pre_trained_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        pre_trained_gpt2.resize_token_embeddings(len(gpt2_tokenizer))
        modules = list(pre_trained_gpt2.children())

        gpt2_linear = modules[1]

        # Init modules
        self.gpt2 = copy.deepcopy(modules[0])
        self.resnet = ResNetEncoder101()
        self.att = LightAttention(self.map_dim, self.hidden_dim)
        self.classifier = gpt2_linear

        self.set_train_on()

    def set_train_on(self, value=True):

        # Activate weight updates
        if value:
            self.gpt2.train()
        else:
            self.gpt2.eval()
        for param in self.gpt2.parameters():
            param.requires_grad = value

        # Activate weight updates
        for param in self.att.parameters():
            param.requires_grad = value

        # Activate weight updates
        for param in self.classifier.parameters():
            param.requires_grad = value

        # Deactivate weight updates
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.show_params()

    def show_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print('Trainable : TRUE ->', name)
            else:
                print('Trainable : FALSE ->', name)

        print('Model parameters: {}'.format(sum(p.numel() for p in self.parameters() if p.requires_grad)))

    def forward(self, sequence, image):
        resnet_maps = self.resnet(image)
        gpt2_hiddens = self.gpt2(sequence)[0]
        out, pixel_softmax_out = self.att(resnet_maps, gpt2_hiddens)
        return self.classifier(out), pixel_softmax_out
