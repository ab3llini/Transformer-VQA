import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

from torch.optim import Adam
from utilities.training.trainer import Trainer
from utilities.paths import resources_path
from datasets.light import LightDataset
from modules.loss import LightLoss
from models.light.model import LightVggGpt2, LightResGpt2, gpt2_tokenizer
import torch


