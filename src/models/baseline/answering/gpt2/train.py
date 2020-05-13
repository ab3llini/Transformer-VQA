import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir, os.pardir, os.pardir))
sys.path.append(root_path)

from utilities.paths import *
from transformers import GPT2LMHeadModel
from utilities.training.legacy_trainer import *
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from datasets.gpt2 import *

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_tokenizer.add_special_tokens(
    {'pad_token': '<pad>', 'bos_token': '<bos>', 'eos_token': '<eos>', 'sep_token': '<sep>'})
X_ENTROPY = CrossEntropyLoss(ignore_index=gpt2_tokenizer.pad_token_id)


def loss_fn(output, labels):
    """
    Loss function calculator
    :param output:
    :param labels:
    :return:
    """
    # Flatten the tensors (shift-align)
    # Remove last token from output
    output = output[..., :-1, :].contiguous().view(-1, output.size(-1))

    # Remove the first token from labels e do not care for question
    labels = (labels[..., 1:].contiguous()).view(-1)

    # Compute the actual loss
    return X_ENTROPY(output, labels)


def logging_fn(out, batch, description):
    ret = ''
    for s in range(3):
        ret += '*' * 25 + '\n'
        ret += '{}'.format(description) + '\n'
        print(batch[0].shape)
        ret += 'Input = {}\n'.format(gpt2_tokenizer.decode(batch[0][s].tolist()))
        ret += 'Output = {}\n'.format(gpt2_tokenizer.decode(torch.argmax(out[0][s], dim=1).tolist()))
    return ret


def train():
    model_basepath = os.path.join('models', 'baseline', 'answering', 'gpt2')
    # Load pre-trained model (weights)
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Resize embeddings by adding the special tokens
    model.resize_token_embeddings(len(gpt2_tokenizer))

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    model.train()

    tr_dataset = GPT2Dataset(directory=resources_path(model_basepath, 'data'),
                             name='training.pk')
    ts_dataset = GPT2Dataset(directory=resources_path(model_basepath, 'data'),
                             name='testing.pk', split='test')

    learning_rate = 5e-5

    gpt2_trainer = LegacyTrainer(
        model=model,
        tr_dataset=tr_dataset,
        ts_dataset=ts_dataset,
        optimizer=Adam(model.parameters(), lr=learning_rate),
        loss=lambda out, batch: loss_fn(out[0], batch[0]),
        lr=learning_rate,
        batch_size=64,
        batch_extractor=lambda batch: batch[1:-1],  # Get rid id & length
        epochs=3,
        tensorboard=SummaryWriter(log_dir=resources_path(model_basepath, 'runs')),
        checkpoint_path=resources_path(model_basepath, 'checkpoints'),
        logging_fp=None,
        logging_fn=logging_fn,
        logging_interval=10
    )

    gpt2_trainer.train()

    del model
    del gpt2_tokenizer
    del gpt2_trainer

if __name__ == '__main__':
    train()
