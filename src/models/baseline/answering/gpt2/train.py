import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir, os.pardir))
sys.path.append(root_path)

from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
from utilities.training import *
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from models.bert import loss as bert_loss
from models.bert.dataset import *
from torch.nn import CrossEntropyLoss


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
    return cross_entropy(output, labels)


def logging_fn(out, batch, description):
    ret = ''
    for s in range(3):
        ret += '*' * 25 + '\n'
        ret += '{}'.format(description) + '\n'
        ret += 'Input = {}\n'.format(tokenizer.decode(batch[0][s].tolist()))
        ret += 'Output = {}\n'.format(tokenizer.decode(torch.argmax(out[0][s], dim=1).tolist()))
    return ret


if __name__ == '__main__':
    model_basepath = os.path.join('models', 'baseline', 'answering', 'gpt2')

    tokenizer = GPT2Tokenizer.from_pretrained()
    tokenizer.add_special_tokens(
        {'pad_token': '<pad>', 'bos_token': '<bos>', 'eos_token': '<eos>', 'sep_token': '<sep>'})

    # Load pre-trained model (weights)
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Resize embeddings by adding the special tokens
    model.resize_token_embeddings(len(tokenizer))

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    model.train()

    cross_entropy = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    tr_dataset = BertDataset(directory=resources_path(model_basepath, 'data'),
                             name='tr_gpt2_answering')
    ts_dataset = BertDataset(directory=resources_path(model_basepath, 'data'),
                             name='ts_gpt2_answering', split='test')

    learning_rate = 5e-5

    bert_trainer = Trainer(
        model=model,
        tr_dataset=tr_dataset,
        ts_dataset=ts_dataset,
        optimizer=Adam(model.parameters(), lr=learning_rate),
        loss=lambda out, batch: bert_loss.loss_fn(out[0], batch[0]),
        lr=learning_rate,
        batch_size=64,
        batch_extractor=lambda batch: [batch[1], batch[3], batch[4]],  # Get rid of the image
        epochs=3,
        tensorboard=SummaryWriter(log_dir=resources_path(model_basepath, 'runs')),
        checkpoint_path=resources_path(model_basepath, 'checkpoints'),
        logging_fp=open(resources_path(model_basepath, 'predictions', 'train.txt'), 'w+'),
        logging_fn=logging_fn,
        logging_interval=10
    )

    bert_trainer.train()
