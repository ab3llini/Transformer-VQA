import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

from models.bert import model
from loaders.vqa import *
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch
from pytorch_transformers import BertConfig
from pytorch_transformers import BertTokenizer
import random



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
decode = lambda text: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

# Settings
batch_size = 64
epochs = 10
logging_interval = 50
learning_rate = 5e-5
dataset_usage = 0.5

# Objects
model = model.Model()
dataset = VQADataset('bert_vgg_padded_types_dataset.pk')
print(dataset.samples[random.randint(0, 2000000)])
loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
optimizer = Adam(model.parameters(), lr=learning_rate)
cross_entropy = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
writer = SummaryWriter()

# Hardware
device = torch.cuda.current_device()
model.to(device)

# Initialize the model
model.zero_grad()

# Reduce the dataset
if dataset_usage < 1:
    print('Dataset usage set to to {}%'.format(dataset_usage * 100))
    print('Shuffling before reducing..')
    random.shuffle(dataset.samples)
    print('Reducing..')
    dataset.samples = dataset.samples[:int(len(dataset.samples) * dataset_usage)]
    print('Done. Total samples = {}'.format(len(dataset.samples)))


def loss_fn(output, labels):
    # Flatten the tensors (shift-align)
    # Remove last token from output
    output = output[..., :-1, :].contiguous().view(-1, output.size(-1))

    # Remove the first token from labels e do not care for question
    labels = (labels[..., 1:].contiguous()).view(-1)

    # Compute the actual loss
    return cross_entropy(output, labels)


iterations = tqdm(loader, desc='Iterating w/ batch size = {}'.format(batch_size))


# Training loop
for epoch in range(epochs):
    for it, batch in enumerate(iterations):

        # Accessing batch objects and moving them to the computing device
        token_ids_masked = batch[0].to(device)
        token_ids = batch[1].to(device)
        token_type_ids = batch[2].to(device)
        attention_mask = batch[3].to(device)
        images = batch[4].to(device)

        # Computing model output
        out = model(token_ids_masked, token_type_ids, attention_mask, images)

        # Compute the loss
        loss = loss_fn(out, token_ids)
        loss.backward()
        optimizer.step()
        model.zero_grad()

        if it % logging_interval == 0:
            for s in range(4):
                print('\nSome predictions..')
                print('Masked input  = ', tokenizer.decode(token_ids_masked[s].tolist()))
                print('Output = ', tokenizer.decode(torch.argmax(out[s], dim=1).tolist()))
                print('Ground truth  = ', tokenizer.decode(token_ids[s].tolist()))


        # Print
        iterations.set_description('Epoch : {}/{} - Loss: {}'.format(epoch + 1, epochs, loss.item()))
        writer.add_scalar('Loss/train', loss.item(), it)

    torch.save(model,
               'bert_vgg_DS%_{}_B_{}_LR_{}_CHKP_EPOCH_{}.h5'.format(dataset_usage, batch_size, learning_rate, epoch))
