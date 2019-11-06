import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir, os.pardir))
sys.path.append(root_path)

from utilities.training.trainer import Trainer
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models.baseline.captioning.model import CaptioningModel
from datasets.captioning import CaptionDataset
from models.baseline.captioning.utils import *
from torch.utils.tensorboard import SummaryWriter

# Model parameters
emb_dim = 256  # dimension of word embeddings
attention_dim = 256  # dimension of attention linear layers
decoder_dim = 256  # dimension of decoder RNN
dropout = 0.4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 10  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 192
workers = 2  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none


def loss_fn(out, batch):
    imgs, scores, caps_sorted, decode_lengths, alphas, sort_ind = out

    # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
    targets = caps_sorted[:, 1:]

    # Remove timesteps that we didn't decode at, or are pads
    # pack_padded_sequence is an easy trick to do this

    scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
    targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

    # Calculate loss
    ce = nn.CrossEntropyLoss()
    loss = ce(scores, targets)

    # Add doubly stochastic attention regularization
    loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

    return loss


def train():
    model_basepath = resources_path('models', 'baseline', 'captioning')

    word_map_file = resources_path(model_basepath, 'data', 'wordmap.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    model = CaptioningModel(attention_dim, emb_dim, decoder_dim, word_map, dropout)

    tr_dataset = CaptionDataset(location=os.path.join(model_basepath, 'data'), split='training')
    ts_dataset = CaptionDataset(location=os.path.join(model_basepath, 'data'), split='testing', maxlen=20000)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print('Trainable : TRUE ->', name)
        else:
            print('Trainable : FALSE ->', name)
    print('Model parameters: {}'.format(sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)))

    caption_trainer = Trainer(
        model=model,
        tr_dataset=tr_dataset,
        ts_dataset=None,
        optimizer=torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=decoder_lr),
        loss=loss_fn,
        lr=decoder_lr,
        batch_size=batch_size,
        device=device,
        epochs=epochs,
        tensorboard=SummaryWriter(log_dir=resources_path(model_basepath, 'runs', 'latest')),
        checkpoint_path=resources_path(model_basepath, 'checkpoints', 'latest')
    )

    caption_trainer.train()

    del model
    del caption_trainer
    del word_map


if __name__ == '__main__':
    train()
