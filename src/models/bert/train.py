import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

from models.bert import model
from utilities.paths import *
from utilities.vqa.dataset import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch
from pytorch_transformers import BertTokenizer
from torch.utils.data.sampler import RandomSampler
import random


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


def test(m, epoch, debug_fp, writer):
    print('Evaluating model.')
    total_loss = 0
    total_batches = 0
    ts_loader = DataLoader(dataset=ts_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)
    with torch.no_grad():
        for it, batch in enumerate(tqdm(ts_loader)):
            # Accessing batch objects and moving them to the computing device
            sequences = batch[1].to(device)
            images = batch[2].to(device)
            token_type_ids = batch[3].to(device)
            attention_masks = batch[4].to(device)

            # Computing model output
            out = m(sequences, token_type_ids, attention_masks, images)

            # Compute the loss
            total_loss += loss_fn(out, sequences).item()
            total_batches += 1

        loss = total_loss / total_batches
        # Visual debug of progress
        writer.add_scalar('Loss/test', loss, epoch)
        debug_fp.write('-' * 25 + '\n')
        debug_fp.write('Test loss :  {} '.format(loss) + '\n')
        debug_fp.write('-' * 25 + '\n')


def train():
    """
    Main training loop.
    """
    # Progress
    global_step = 0

    for epoch in range(epochs):

        iterations = tqdm(tr_loader)

        for it, batch in enumerate(iterations):

            # Accessing batch objects and moving them to the computing device
            sequences = batch[1].to(device)
            images = batch[2].to(device)
            token_type_ids = batch[3].to(device)
            attention_masks = batch[4].to(device)

            # Computing model output
            out = model(sequences, token_type_ids, attention_masks, images)

            # Compute the loss
            loss = loss_fn(out, sequences)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if it % logging_interval == 0:
                for s in range(n_prediction_samples):
                    debug_fp.write('*' * 25 + '\n')
                    debug_fp.write('Epoch {} - Iteration {}'.format(epoch, it) + '\n')
                    debug_fp.write('Input = {}\n'.format(tokenizer.decode(sequences[s].tolist())))
                    debug_fp.write('Output = {}\n'.format(tokenizer.decode(torch.argmax(out[s], dim=1).tolist())))

            # Visual debug of progress
            iterations.set_description('Epoch : {}/{} - Loss: {}'.format(epoch + 1, epochs, loss.item()))
            writer.add_scalar('Loss/train', loss.item(), global_step)

            global_step += 1

        torch.save(model.state_dict(), resources_path('models', 'bert', 'checkpoints',
                                                      '{}_B_{}_LR_{}_CHKP_EPOCH_{}.pth'.format(model_name, batch_size,
                                                                                              learning_rate, epoch)))

        # Evaluate test loss
        test(model, epoch, debug_fp, writer)


if __name__ == '__main__':

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    decode = lambda text: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

    # Settings
    model_name = 'bert_vgg_1M'
    batch_size = 64
    epochs = 10
    logging_interval = 50
    learning_rate = 5e-5
    n_prediction_samples = 3
    prediction_file_name = resources_path('models', 'bert', 'train_predictions', model_name + '.txt')

    # Debugging
    debug_fp = open(prediction_file_name, 'w+')

    # Objects
    model = model.Model()
    tr_dataset = BertDataset(directory=resources_path('models', 'bert', 'data'), name='tr_bert_1M.pk')
    ts_dataset = BertDataset(directory=resources_path('models', 'bert', 'data'), name='ts_bert_1M.pk', maxlen=2000)

    tr_loader = DataLoader(dataset=tr_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=2)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    cross_entropy = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    writer = SummaryWriter(log_dir=resources_path('models', 'bert', 'runs'))

    # Hardware
    device = torch.cuda.current_device()
    model.to(device)

    # Initialize the model
    model.zero_grad()
    train()
