import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir))
sys.path.append(root_path)

import torch
from utilities.vqa.dataset import *
from transformers import GPT2Tokenizer
from datasets.creator import DatasetCreator, MultiPurposeDataset
from utilities.evaluation.beam_search import BeamSearchInput

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_tokenizer.add_special_tokens(
    {'pad_token': '<pad>', 'bos_token': '<bos>', 'eos_token': '<eos>', 'sep_token': '<sep>'})


class ResGPT2Dataset(MultiPurposeDataset):

    def __getitem__(self, item):

        sample = self.data[item]

        sample = self.data[item]
        if not self.evaluating:
            _, sequence, image_path = sample
        else:
            __id, question, image_path = sample

        image = load_image(image_rel_path=image_path)
        resized_image = resize_image(image)
        image = normalized_tensor_image(resized_image)

        if not self.evaluating:
            return torch.tensor(sequence).long(), \
                   image
        else:
            question = torch.tensor(question).long()
            beam_input = BeamSearchInput(0, 0, question, image)
            ground_truths = self.evaluation_data[str(__id)]
            return __id, beam_input, ground_truths, resized_image
