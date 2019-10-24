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


def create_datasets(base_path):
    longest_answer = {'training': 0, 'testing': 0}

    def elem_processing_fn(question_id, question, image_path, answer, split):
        question_tkn = gpt2_tokenizer.encode(question)
        answer_tkn = gpt2_tokenizer.encode(answer)
        question_tkn_len = len(question_tkn)
        answer_tkn_len = len(answer_tkn)

        if longest_answer[split] < answer_tkn_len:
            longest_answer[split] = answer_tkn_len

        if longest_answer[split] < answer_tkn_len + question_tkn_len:
            longest_answer[split] = answer_tkn_len + question_tkn_len

        # Add BOS, SEP & EOS tokens
        question_tkn = [gpt2_tokenizer.bos_token_id] + question_tkn + [gpt2_tokenizer.sep_token_id]
        answer_tkn = answer_tkn + [gpt2_tokenizer.eos_token_id]

        # Concatenate Q+A
        sequence = question_tkn + answer_tkn

        return [question_id, sequence, question_tkn_len + 2]

    def post_processing_fn(tr_data, ts_data):

        print('Longest = {}'.format(longest_answer))

        # Pad sequences
        print('Padding training sequences..')
        tr_data = DatasetCreator.pad_sequences(tr_data, axis=1, value=int(gpt2_tokenizer.pad_token_id),
                                               maxlen=longest_answer['training'] + 3)

        # Pad sequences
        print('Padding testing sequences..')
        ts_data = DatasetCreator.pad_sequences(ts_data, axis=1, value=int(gpt2_tokenizer.pad_token_id),
                                               maxlen=longest_answer['testing'] + 3)

        return tr_data, ts_data

    DatasetCreator().create_together(tr_size=1000000, ts_size=200000,
                                     tr_destination=os.path.join(base_path, 'training'),
                                     ts_destination=os.path.join(base_path, 'testing'),
                                     elem_processing_fn=elem_processing_fn, post_processing_fn=post_processing_fn)


class GPT2Dataset(MultiPurposeDataset):

    def __getitem__(self, item):

        sample = self.data[item]
        if not self.evaluating:
            _, sequence, _ = sample
        else:
            __id, sequence, ql = sample

        if not self.evaluating:
            # Return answer + image + length
            return torch.tensor(sequence).long()
        else:
            question = torch.tensor(sequence[:ql]).long()
            beam_input = BeamSearchInput(0, 0, question)
            ground_truths = self.evaluation_data[str(__id)]
            return beam_input, ground_truths


if __name__ == '__main__':
    path = resources_path('models', 'baseline', 'answering', 'gpt2', 'data')
    create_datasets(path)
