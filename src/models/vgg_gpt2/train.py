import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

import torch
import skimage.transform
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from utilities.training.trainer import Trainer
from utilities.paths import resources_path
from datasets.vgg_gpt2 import VGGPT2Dataset
from modules.loss import GPT2Loss
from transformers import GPT2Tokenizer
from models.vgg_gpt2.model import VGGPT2
from utilities.training.logger import transformer_output_log
from torchvision import transforms
from PIL import Image
from utilities.vqa.dataset import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import random

q_path_tr, a_path_tr, i_path_tr = get_data_paths(data_type='train')
vqa_helper_tr = VQA(a_path_tr, q_path_tr)

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_tokenizer.add_special_tokens(
    {'pad_token': '<pad>', 'bos_token': '<bos>', 'eos_token': '<eos>', 'sep_token': '<sep>'})


def softmap_visualize(softmaps, sequence, image, i_path):
    """
    Visualize attention maps over the image
    :return: The softmaps as PIL objects
    """

    softmaps = softmaps.detach().to('cpu')
    image = load_image(i_path, image)

    image = image.resize([7 * 32, 7 * 32], Image.LANCZOS)
    words_tokenized = sequence.tolist()
    words = [gpt2_tokenizer.convert_ids_to_tokens(w) for w in words_tokenized]
    softmaps = softmaps.view(softmaps.size(0), 7, 7)
    words = words[:words.index('<pad>')]

    for t in range(len(words)):
        if t > 25:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)
        if words[t][0] == 'Ġ':  # Remove new word indicator of gpt2 tokenizer
            words[t] = words[t][1:]

        plt.text(0, 1, '%s' % words[t], color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = softmaps[t, :]
        # alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=32, sigma=8)

        alpha = skimage.transform.resize(current_alpha.numpy(), [7 * 32, 7 * 32])
        if words[t] in ['<bos>', '<eos>', '<sep>']:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        # plt.set_cmap(cm.Greys_r)
        plt.axis('off')

    fig = plt.gcf()
    # plt.show()

    return fig, words


def gpt2_callback_fn(output, batch, iteration, epoch, task, tb):
    softmap_fig, words = softmap_visualize(
        softmaps=output[1][0],
        sequence=batch[1][0],
        image=vqa_helper_tr.getImgIds(quesIds=[batch[0][0].item()])[0],
        i_path=i_path_tr
    )

    tb.add_figure(
        tag='IT_{}_ID_{}_{}'.format(iteration, batch[0][0].item(), task.upper()),
        figure=softmap_fig,
        global_step=epoch
    )


def train():
    loss = GPT2Loss(pad_token_id=gpt2_tokenizer.pad_token_id)
    model = VGGPT2(tokenizer=gpt2_tokenizer)

    model_basepath = os.path.join('models', 'vgg_gpt2')

    tr_dataset = VGGPT2Dataset(directory=resources_path(model_basepath, 'data'),
                               name='training.pk')

    learning_rate = 5e-5
    epochs = 20

    tb = SummaryWriter(log_dir=resources_path(model_basepath, 'runs', 'exp2'))

    gpt2_trainer = Trainer(
        model=model,
        tr_dataset=tr_dataset,
        ts_dataset=None,
        optimizer=Adam(model.parameters(), lr=learning_rate),
        loss=lambda out, batch: loss(out[0], batch[0]),
        lr=learning_rate,
        batch_size=40,
        batch_extractor=lambda batch: batch[1:],  # Get rid id & original image
        epochs=epochs,
        tensorboard=tb,
        checkpoint_path=resources_path(model_basepath, 'checkpoints'),
        callback_fn=lambda *args: gpt2_callback_fn(*(list(args) + [tb])),
        callback_interval=50,
        device='cuda'
    )

    gpt2_trainer.train()


if __name__ == '__main__':
    train()
