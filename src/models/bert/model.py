import torch.nn as nn
from torchvision import models
import copy
from pytorch_transformers import BertForMaskedLM, BertConfig


class ImageEncoder(nn.Module):

    def __init__(self, out_size=768, verbose=False):
        super(ImageEncoder, self).__init__()

        # Using pre-trained vgg
        vgg = models.vgg11(pretrained=True)

        # Remove the classifier
        modules = list(vgg.children())[:-1]

        # Keep only the network
        self.vgg = nn.Sequential(*modules)

        # Make sure to disable weight update
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, images):
        # outputs maps from vgg (512 channels average pooled)
        out = self.vgg(images)  # (batch_size, 512, 7, 7)
        return out.permute(0, 2, 3, 1)  # (batch_size, 7, 7, 512)


class Attention(nn.Module):

    def __init__(self, image_channels=512, bert_size=768, attention_dim=512):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(image_channels, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(bert_size, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, vgg_out, bert_hidden):
        """
        Forward propagation.
        :param vgg_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param bert_hidden: BERT hidden state, a tensor of dimension (batch_size, n_tokens, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(vgg_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(bert_hidden)  # (batch_size, n_token, attention_dim)
        att = self.full_att(self.relu(att1.unsqueeze(1) + att2.unsqueeze(2))).squeeze(
            3)  # (batch_size, n_tokens, num_pixels)
        alpha = self.softmax(att)  # (batch_size, n_tokens, num_pixels)
        attention_weighted_encoding = (vgg_out.unsqueeze(1) * alpha.unsqueeze(3)).sum(
            dim=2)  # (batch_size,n_token, encoder_dim)

        return attention_weighted_encoding, alpha


# This is the BERT model
class LanguageModel(nn.Module):
    def __init__(self, bert):
        super(LanguageModel, self).__init__()

        self.bert = bert

        # Make sure to disable weight update
        for param in self.bert.parameters():
            param.requires_grad = False

    # Standard forward, make sure to prepare inputs properly
    def forward(self, token_ids, token_type_ids, attention_mask):
        return self.bert(token_ids, token_type_ids, attention_mask)[0]  # (batch_size, n_tokens, bert_size)


# Multimodal fusion
class Model(nn.Module):
    def __init__(self, attention_dim=512):
        super(Model, self).__init__()

        full_bert = BertForMaskedLM.from_pretrained('bert-base-uncased')
        modules = list(full_bert.children())
        bert = copy.deepcopy(modules[0])
        head = copy.deepcopy(list((list(modules[1].children())[0]).modules())[-1])

        self.language_model = LanguageModel(bert=bert)
        self.image_encoder = ImageEncoder()
        self.question_image_att = Attention(attention_dim=attention_dim)
        self.attention_fc = nn.Linear(attention_dim, 768)  # 768 is the size of the bert FC last layer
        self.classifier = head

        # Activate weight update for the last FC layer
        for param in self.classifier.parameters():
            param.requires_grad = True

        for name, param in self.named_parameters():
            if param.requires_grad:
                print('Trainable : TRUE ->', name)
            else:
                print('Trainable : FALSE ->', name)

        print(f'Model parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}')

    def forward(self, token_ids, token_type_ids, attenton_mask, images):

        language_model_out = self.language_model(token_ids, token_type_ids, attenton_mask)  # (batch_size, n_tokens, bert_size)
        image_encoder_out = self.image_encoder(images)  # (batch_size, 7, 7, encoder_dim)

        batch_size = image_encoder_out.size(0)
        image_encoder_dim = image_encoder_out.size(-1)

        # Flatten the image encoder output on the pixel dimension
        image_encoder_out = image_encoder_out.view(batch_size, -1, image_encoder_dim)  # attention, alphas

        # Compute the attention over the 512 channels of VGG using the bert hidden hyper parameters
        attention_out = self.question_image_att(image_encoder_out, language_model_out)[0]  # (batch_size, n_tokens, encoder_dim)

        # Bring the attention output to the same space of the bert output, then multiply
        fusion = language_model_out * self.attention_fc(attention_out)  # (batch_size, n_tokens, bert_size)

        # Final FC model head to distribute over the vocab.
        predictions = self.classifier(fusion)

        return predictions
