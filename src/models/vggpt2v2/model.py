import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

from torch import nn
from modules.image_encoders import *
from torchvision import models
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from modules.attention import CoAttention
from modules.mm import ModularGpt2
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


class VGGPTv2(ModularGpt2):
    def __init__(self):
        super(VGGPTv2, self).__init__()

        # Image encoder
        self.image_encoder = VGGEncoder(models.vgg19(pretrained=True))
        # Co-Attention layer
        self.co_att = CoAttention(512, 768, 512)
        # Classifier
        self.classifier = nn.Linear(in_features=768 * 2, out_features=len(gpt2_tokenizer))

        # Copy the original weights and concat the new ones for the attention
        with torch.no_grad():
            self.classifier.weight.copy_(
                torch.cat(
                    [
                        self.head.weight,
                        torch.zeros(self.head.weight.size())
                    ],
                    dim=1
                )
            )

        del self.head

        # Disable weight update for both VGG and GPT-2
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.gpt2.parameters():
            p.requires_grad = False

        # Enable weight updates for classifier and attention layer
        for p in self.classifier.parameters():
            p.requires_grad = True
        for p in self.co_att.parameters():
            p.requires_grad = True

    def forward(self, sequence, image):
        maps = self.image_encoder(image)
        hiddens = self.gpt2(sequence)[0]
        co_att_out, pixel_softmax_out = self.co_att(maps, hiddens)
        concat = torch.cat([hiddens, co_att_out], dim=2)
        return self.classifier(concat), pixel_softmax_out


if __name__ == '__main__':
    VGGPTv2().show_params()
