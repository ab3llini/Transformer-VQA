from torch import nn
import torch
from torchvision import models


class VGGEncoder11(nn.Module):

    def __init__(self):
        super(VGGEncoder11, self).__init__()

        # Using pre-trained vgg
        vgg = models.vgg11(pretrained=True)

        # Remove the classifier
        modules = list(vgg.children())[:-1]

        # Keep only the network
        self.vgg = nn.Sequential(*modules)

    def forward(self, images):
        # outputs maps from vgg (512 channels average pooled @ 7x7)
        out = self.vgg(images)  # (batch_size, 512, 7, 7)
        return out.permute(0, 2, 3, 1)  # (batch_size, 7, 7, 512)


if __name__ == '__main__':
    x = VGGEncoder11()
    y = list(x.children())[0][0][-1]
    print(y)
