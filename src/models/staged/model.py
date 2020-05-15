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
from datasets.vggpt2v2 import *

# Import of what we will use
from modules.mm import ModularGpt2
from modules.loss import GPT2Loss, VisualGPT2Loss
from utilities.training.trainer import Trainer
from modules.image_encoders import *
from utilities.evaluation import predict as pred

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


def stage_one():
    basepath = os.path.join('models', 'baseline', 'answering', 'gpt2')

    model = ModularGpt2(emd_size=len(gpt2_tokenizer))

    tr_dataset = GPT2Dataset(resources_path(os.path.join(basepath, 'data')))
    ts_dataset = GPT2Dataset(resources_path(os.path.join(basepath, 'data')), split='testing')

    learning_rate = 5e-5
    epochs = 10
    batch_size = 128
    early_stopping = None

    loss = GPT2Loss(
        pad_token_id=gpt2_tokenizer._convert_token_to_id('<pad>')
    )

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


class StageTwo(ModularGpt2):
    def __init__(self, stage_one_checkpoint):
        # Initialize a stage one model
        super(StageTwo, self).__init__(emd_size=len(gpt2_tokenizer))
        # Initialize from checkpoint
        self.load_state_dict(torch.load(stage_one_checkpoint))
        # Add Image Encoder
        self.image_encoder = ResNetEncoder(encoded_image_size=14, instance=models.resnet152(pretrained=True))
        # Linear compression (from 2048 to 768)
        self.linear = nn.Linear(in_features=2048, out_features=768)

    def forward(self, sequence, image):
        # (Batch size, 192, 2048)
        maps = self.image_encoder(image).reshape(-1, 14 * 14, 2048)
        # (Batch size, 1, 2048)
        attention = maps.mean(dim=1)
        # (Batch size, 1, 768)
        attention = self.linear(attention).unsqueeze(1)
        # (Batch size, sequence length, 768)
        hiddens = self.gpt2(sequence)[0]
        # (Batch size, sequence length, 768)
        pointwise = hiddens * attention
        # (Batch size, sequence length, voc_size)
        out = self.head(pointwise)

        return out


def explain(model, checkpoint_n, checkpoint_p, n_checkpoints, *args):
    # Reproducibility
    torch.manual_seed(0)

    print(f'Model output without any checkpoint')
    answer = pred.answer(model, [gpt2_tokenizer.eos_token_id], 20, 'cuda', gpt2_tokenizer, *args)
    print(pred.pretty_format(args[0], answer, gpt2_tokenizer))

    # Checkpoint iterations
    for i in range(n_checkpoints):
        name = checkpoint_n.format(i)
        checkpoint = os.path.join(checkpoint_p, name)
        model.load_state_dict(torch.load(checkpoint))
        print(f'Model loaded and initialized with checkpoint = {name}')
        answer = pred.answer(model, [gpt2_tokenizer.eos_token_id], 20, 'cuda', gpt2_tokenizer, *args)
        print(pred.pretty_format(args[0], answer, gpt2_tokenizer))


def stage_two(checkpoint_n):
    # Load stage one checkpoint
    checkpoint_n = f'ModularGpt2_bs=128_lr=5e-05_e={checkpoint_n}.pth'
    checkpoint_p = resources_path('models', 'baseline', 'answering', 'gpt2', 'checkpoints', 'latest')
    checkpoint = os.path.join(checkpoint_p, checkpoint_n)

    model = StageTwo(stage_one_checkpoint=checkpoint)

    bp = resources_path('models', 'vggpt2v2')

    tr_dataset = VGGPT2v2Dataset(resources_path(os.path.join(bp, 'data')))
    ts_dataset = VGGPT2v2Dataset(resources_path(os.path.join(bp, 'data')), split='testing')

    learning_rate = 5e-5
    epochs = 10
    batch_size = 128
    early_stopping = None

    loss = VisualGPT2Loss(
        pad_token_id=gpt2_tokenizer._convert_token_to_id('<pad>')
    )

    trainer = Trainer(
        wandb_args={'project': 'multi-stage-vqa', 'name': 'stage-two'},
        model=model,
        tr_dataset=tr_dataset,
        ts_dataset=ts_dataset,
        optimizer=Adam(model.parameters(), lr=learning_rate),
        loss=loss,
        epochs=epochs,
        early_stopping=early_stopping,
        num_workers=4,
        checkpoint_path=resources_path(bp, 'checkpoints', 'latest'),
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
    # stage_one()
    # explain(
    #     ModularGpt2(emd_size=len(gpt2_tokenizer)),
    #     'ModularGpt2_bs=128_lr=5e-05_e={}.pth',
    #     resources_path('models', 'baseline', 'answering', 'gpt2', 'checkpoints', 'latest'),
    #     10,
    #     'It was a rainy day when suddenly '
    # )
    stage_two(7)
