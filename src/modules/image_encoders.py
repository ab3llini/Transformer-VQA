import torchvision
from torch import nn
import torch
from torchvision import models


class VGGEncoder(nn.Module):

    def __init__(self, instance=models.vgg11(pretrained=True)):
        super(VGGEncoder, self).__init__()

        # Using pre-trained vgg
        vgg = instance

        # Remove the classifier
        modules = list(vgg.children())[:-1]

        # Keep only the network
        self.vgg = nn.Sequential(*modules)

    def forward(self, images):
        # outputs maps from vgg (512 channels average pooled @ 7x7)
        out = self.vgg(images)  # (batch_size, 512, 7, 7)
        return out.permute(0, 2, 3, 1)  # (batch_size, 7, 7, 512)


class ResNetEncoder101(nn.Module):

    def __init__(self, encoded_image_size=14):
        super(ResNetEncoder101, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        for p in self.resnet.parameters():
            p.requires_grad = False

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

