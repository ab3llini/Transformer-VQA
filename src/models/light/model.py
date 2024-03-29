import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

from torch import nn
from modules.image_encoders import *
from torchvision import models
from transformers import GPT2LMHeadModel, GPT2Tokenizer

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


class ModularGpt2(nn.Module):
    def __init__(self):
        super(ModularGpt2, self).__init__()

        gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        self.head = list(gpt2.children())[1]
        self.gpt2 = list(gpt2.children())[0]

    def show_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print('Trainable : TRUE ->', name)
            else:
                print('Trainable : FALSE ->', name)

        print('Trainable parameters: {}'.format(sum(p.numel() for p in self.parameters() if p.requires_grad)))
        print('Total parameters: {}'.format(sum(p.numel() for p in self.parameters())))


class LightVggGpt2(ModularGpt2):
    def __init__(self):
        super(LightVggGpt2, self).__init__()

        # Image encoder
        self.image_encoder = VGGEncoder(models.vgg19(pretrained=True))
        # Linear expansion (from 512 to 768)
        self.expansion = nn.Linear(in_features=512, out_features=768)

        # Disable weight update for both VGG and GPT-2
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.gpt2.parameters():
            p.requires_grad = False

        # Enable weight updates for GPT-2 head and expander
        for p in self.expansion.parameters():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

            # 1 implement E S valdation loss
            # max limit # epochs to 100
            # train cuncurr 2 model with vgg
            # 1 model avg
            # 2 model max

    def forward(self, sequence, image):
        # (Batch size, 49, 512)
        maps = self.image_encoder(image).reshape(-1, 49, 512)
        # (Batch size, 1, 49, 768)
        maps = self.expansion(maps).unsqueeze(1)
        # (Batch size, sequence length, 1, 768)
        hiddens = self.gpt2(sequence)[0].unsqueeze(2)
        # (Batch size, sequence length, 768)
        combined = (maps + hiddens).sum(dim=2).squeeze(2)
        # (Batch size, sequence length, voc_size)
        out = self.head(combined)

        return out


class LightResGpt2(ModularGpt2):
    def __init__(self):
        super(LightResGpt2, self).__init__()
        # Image encoder
        self.image_encoder = ResNetEncoder(encoded_image_size=14, instance=models.resnet152(pretrained=True))
        # Linear expansion (from 512 to 768)
        self.expansion = nn.Linear(in_features=2048, out_features=768)

        # Disable weight update for both VGG and GPT-2
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.gpt2.parameters():
            p.requires_grad = False

        # Enable weight updates for GPT-2 head and expander
        for p in self.expansion.parameters():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def forward(self, sequence, image):

        # (Batch size, 192, 2048)
        maps = self.image_encoder(image).reshape(-1, 14 * 14, 2048)
        # (Batch size, 1, 192, 768)
        maps = self.expansion(maps).unsqueeze(1)
        # (Batch size, sequence length, 1, 768)
        hiddens = self.gpt2(sequence)[0].unsqueeze(2)
        # (Batch size, sequence length, 768)
        combined = (maps + hiddens).sum(dim=2).squeeze(2)
        # (Batch size, sequence length, voc_size)
        out = self.head(combined)

        return out


class LightVggGpt2Avg(ModularGpt2):
    def __init__(self):
        super(LightVggGpt2Avg, self).__init__()

        # Image encoder
        self.image_encoder = VGGEncoder(models.vgg19(pretrained=True))
        # Linear expansion (from 512 to 768)
        self.expansion = nn.Linear(in_features=512, out_features=768)
        # self.expansion.weight.data = torch.zeros(self.expansion.weight.size())

        # Disable weight update for both VGG and GPT-2
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.gpt2.parameters():
            p.requires_grad = False

        # Enable weight updates for GPT-2 head and expander
        for p in self.expansion.parameters():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = False

            # 1 implement E S valdation loss
            # max limit # epochs to 100
            # train cuncurr 2 model with vgg
            # 1 model avg
            # 2 model max

    def forward(self, sequence, image):
        # (Batch size, 512)
        maps = self.image_encoder(image).reshape(-1, 49, 512).mean(dim=1)
        # (Batch size, 1, 768)
        maps = self.expansion(maps).unsqueeze(1)
        # (Batch size, sequence length, 768)
        hiddens = self.gpt2(sequence)[0]
        # (Batch size, sequence length, 768)
        combined = (maps + hiddens)
        # (Batch size, sequence length, voc_size)
        out = self.head(combined)

        return out


class LightVggGpt2Max(ModularGpt2):
    def __init__(self):
        super(LightVggGpt2Max, self).__init__()

        # Image encoder
        self.image_encoder = VGGEncoder(models.vgg19(pretrained=True))
        # Linear expansion (from 512 to 768)
        self.expansion = nn.Linear(in_features=512, out_features=768)
        # self.expansion.weight.data = torch.zeros(self.expansion.weight.size())

        # Disable weight update for both VGG and GPT-2
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.gpt2.parameters():
            p.requires_grad = False

        # Enable weight updates for GPT-2 head and expander
        for p in self.expansion.parameters():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = False

    def forward(self, sequence, image):
        # (Batch size, 512)
        maps = self.image_encoder(image).reshape(-1, 49, 512).max(dim=1)[0]
        # (Batch size, 1, 768)
        maps = self.expansion(maps).unsqueeze(1)
        # (Batch size, sequence length, 768)
        hiddens = self.gpt2(sequence)[0]
        # (Batch size, sequence length, 768)
        combined = (maps + hiddens)
        # (Batch size, sequence length, voc_size)
        out = self.head(combined)

        return out


class LightVggGpt2AvgConcat(ModularGpt2):
    def __init__(self):
        super(LightVggGpt2AvgConcat, self).__init__()

        # Image encoder
        self.image_encoder = VGGEncoder(models.vgg19(pretrained=True))
        # Linear expansion (from 512 to 768)
        self.expansion = nn.Linear(in_features=512, out_features=768)
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

        # Enable weight updates for GPT-2 head and expander
        for p in self.expansion.parameters():
            p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True

    def forward(self, sequence, image):
        # (Batch size, 512)
        maps = self.image_encoder(image).reshape(-1, 49, 512).mean(dim=1)
        # (Batch size, sequence length, 768)
        hiddens = self.gpt2(sequence)[0]
        # (Batch size, sequence length, 768)
        maps = self.expansion(maps).unsqueeze(1).expand(-1, hiddens.shape[1], -1)
        # (Batch size, sequence length, 768)
        concat = torch.cat([hiddens, maps], dim=2)
        # (Batch size, sequence length, voc_size)
        out = self.classifier(concat)

        return out


class LightVggGpt2MaxConcat(ModularGpt2):
    def __init__(self):
        super(LightVggGpt2MaxConcat, self).__init__()

        # Image encoder
        self.image_encoder = VGGEncoder(models.vgg19(pretrained=True))
        # Linear expansion (from 512 to 768)
        self.expansion = nn.Linear(in_features=512, out_features=768)
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

        # Enable weight updates for GPT-2 head and expander
        for p in self.expansion.parameters():
            p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True

    def forward(self, sequence, image):
        # (Batch size, 512)
        maps = self.image_encoder(image).reshape(-1, 49, 512).max(dim=1)[0]
        # (Batch size, sequence length, 768)
        hiddens = self.gpt2(sequence)[0]
        # (Batch size, sequence length, 768)
        maps = self.expansion(maps).unsqueeze(1).expand(-1, hiddens.shape[1], -1)
        # (Batch size, sequence length, 768)
        concat = torch.cat([hiddens, maps], dim=2)
        # (Batch size, sequence length, voc_size)
        out = self.classifier(concat)

        return out


if __name__ == '__main__':
    """
    LightResGpt2().show_params()
    LightVggGpt2().show_params()
    LightVggGpt2Avg().show_params()
    LightVggGpt2Max().show_params()
    """
    LightVggGpt2AvgConcat().show_params()
