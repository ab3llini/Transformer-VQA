import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

from utilities.paths import *
from transformers import GPT2LMHeadModel
from utilities.training.legacy_trainer import *
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from datasets.gpt2 import *

# Import of what we will use
from modules.mm import ModularGpt2
from modules.loss import GPT2Loss
from utilities.training.trainer import Trainer

# Goal of this model

# 1) train the transformer WITHOUT the images to perform QA. Do it the same as you were doing it previously -- either on
# a big QA dataset or on the VQA dataset without the V
#
# 2) STARTING from the model above, add the images only WITHOUT
# ATTENTION, (https://docs.google.com/drawings/d/1Aca29E7Chtw293lc3UPLXdiwRD6AcmVTdruHeNleDSA/edit?usp=sharing)
#
# 3) STARTING from the model above, add the MULTI-HEAD ATTENTION using the SAME LINEAR MAPPING (which is
# fine-tuned but tied) as shown here:

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens(
    {'pad_token': '<pad>', 'bos_token': '<bos>', 'eos_token': '<eos>', 'sep_token': '<sep>'}
)


def stage_one() -> ModularGpt2:
    basepath = os.path.join('models', 'baseline', 'answering', 'gpt2')

    loss = GPT2Loss(
        pad_token_id=gpt2_tokenizer._convert_token_to_id('<pad>')
    )

    model = ModularGpt2(emd_size=len(gpt2_tokenizer))

    tr_dataset = GPT2Dataset(resources_path(os.path.join(basepath, 'data')))
    ts_dataset = GPT2Dataset(resources_path(os.path.join(basepath, 'data')), split='testing')

    learning_rate = 5e-5
    epochs = 10
    batch_size = 64
    early_stopping = None

    trainer = Trainer(
        wandb_args={'project': 'multi-stage-vqa', 'name': 'stage-one'},
        model=model,
        tr_dataset=tr_dataset,
        ts_dataset=ts_dataset,
        optimizer=Adam(model.parameters(), lr=learning_rate),
        loss=loss,
        epochs=epochs,
        early_stopping=early_stopping,
        num_workers=4,
        checkpoint_path=resources_path(basepath, 'checkpoints', 'latest'),
        load_checkpoint=None,
        device='cuda',
        shuffle=True,
        log_interval=10,
        lr=learning_rate,
        batch_size=batch_size
    )

    trainer.run()


if __name__ == '__main__':
    # Pipeline
    # Train the QA Model first
    stage_one()
