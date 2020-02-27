import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)


from torch import nn

import copy
from modules.image_encoders import VGGEncoder
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from modules.attention import CoAttention
import torch

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_tokenizer.add_special_tokens(
    {'pad_token': '<pad>', 'bos_token': '<bos>', 'eos_token': '<eos>', 'sep_token': '<sep>'})


class VGGPT2(nn.Module):
    def __init__(self, attention_dim=512):
        super(VGGPT2, self).__init__()

        self.attention_dim = attention_dim
        self.map_dim = 512  # Number of VGG11 channels
        self.hidden_dim = 768  # Number of hidden parameters in GPT2 (small)

        # Resize the language model embedding layer

        pre_trained_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        pre_trained_gpt2.resize_token_embeddings(len(gpt2_tokenizer))
        modules = list(pre_trained_gpt2.children())

        gpt2_linear = modules[1].weight
        attention_linear = torch.zeros(gpt2_linear.size())
        print(attention_linear.shape)

        # Init modules
        self.gpt2 = copy.deepcopy(modules[0])
        self.vgg11 = VGGEncoder()
        self.co_att = CoAttention(self.map_dim, self.hidden_dim, self.attention_dim)

        self.classifier = nn.Linear(in_features=self.hidden_dim * 2, out_features=len(gpt2_tokenizer))
        # Copy the original weights and concat the new ones for the attention
        with torch.no_grad():

            self.classifier.weight.copy_(torch.cat([gpt2_linear, attention_linear], dim=1))

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
        for param in self.co_att.parameters():
            param.requires_grad = value

        # Activate weight updates
        for param in self.classifier.parameters():
            param.requires_grad = value

        # Deactivate weight updates
        if value:
            self.vgg11.vgg.train()
        else:
            self.vgg11.vgg.eval()
        for param in self.vgg11.parameters():
            param.requires_grad = False

        # self.show_params()

    def show_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print('Trainable : TRUE ->', name)
            else:
                print('Trainable : FALSE ->', name)

        print('Model parameters: {}'.format(sum(p.numel() for p in self.parameters() if p.requires_grad)))

    def forward(self, sequence, image):
        vgg_maps = self.vgg11(image)
        gpt2_hiddens = self.gpt2(sequence)[0]
        co_att_out, pixel_softmax_out = self.co_att(vgg_maps, gpt2_hiddens)
        concat = torch.cat([gpt2_hiddens, co_att_out], dim=2)
        return self.classifier(concat), pixel_softmax_out

if __name__ == '__main__':
    VGGPT2()