from torch import nn
from transformers import GPT2LMHeadModel


class ModularGpt2(nn.Module):
    def __init__(self, emd_size=None):
        super(ModularGpt2, self).__init__()

        gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        if emd_size:
            gpt2.resize_token_embeddings(emd_size)

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

    def forward(self, sequence):
        return self.head(self.gpt2(sequence)[0])