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
from collections import Counter


gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_tokenizer.add_special_tokens(
    {'pad_token': '<pad>', 'bos_token': '<bos>', 'eos_token': '<eos>', 'sep_token': '<sep>'})


def create_datasets(base_path):
    seq_counter = {
        'training': Counter(),
        'testing': Counter()
    }

    def elem_processing_fn(question_id, question, image_path, answer, split):
        question_tkn = gpt2_tokenizer.encode(question)
        answer_tkn = gpt2_tokenizer.encode(answer)
        question_tkn_len = len(question_tkn)
        answer_tkn_len = len(answer_tkn)

        seq_counter[split].update([answer_tkn_len + question_tkn_len])


        # Add BOS, SEP & EOS tokens
        question_tkn = [gpt2_tokenizer.bos_token_id] + question_tkn + [gpt2_tokenizer.sep_token_id]
        answer_tkn = answer_tkn + [gpt2_tokenizer.eos_token_id]

        # Concatenate Q+A
        sequence = question_tkn + answer_tkn

        return [question_id, sequence, question_tkn_len + 2]

    def post_processing_fn(tr_data, ts_data, tr_size, ts_size):

        tr_removed = len(tr_data)
        ts_removed = len(ts_data)

        print('Removing short samples checking frequencies')
        tr_data = list(filter(lambda item: seq_counter['training'][len(item[1])] > 1000, tr_data))
        ts_data = list(filter(lambda item: seq_counter['testing'][len(item[1])] > 1000, ts_data))

        print(seq_counter)

        tr_removed -= len(tr_data)
        ts_removed -= len(ts_data)

        print('Removed {} from training data and {} from testing data'.format(tr_removed, ts_removed))

        tr_data = tr_data[:tr_size]
        ts_data = ts_data[:ts_size]

        print('Len tr = {}, len ts = {}'.format(len(tr_data), len(ts_data)))

        max_len_tr = 0
        max_len_ts = 0

        for length, freq in seq_counter['training'].items():
            if freq > 1000:
                if max_len_tr < length:
                    max_len_tr = length

        for length, freq in seq_counter['testing'].items():
            if freq > 1000:
                if max_len_ts < length:
                    max_len_ts = length


        # Pad sequences
        print('Padding training sequences..')
        tr_data = DatasetCreator.pad_sequences(tr_data, axis=1, value=int(gpt2_tokenizer.pad_token_id),
                                               maxlen=max_len_tr + 3)

        # Pad sequences
        print('Padding testing sequences..')
        ts_data = DatasetCreator.pad_sequences(ts_data, axis=1, value=int(gpt2_tokenizer.pad_token_id),
                                               maxlen=max_len_ts + 3)

        return tr_data, ts_data

    DatasetCreator().create_together(tr_size=1000000, ts_size=100000,
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
            return __id, beam_input, ground_truths


if __name__ == '__main__':
    path = resources_path('models', 'baseline', 'answering', 'gpt2', 'data')
    create_datasets(path)
