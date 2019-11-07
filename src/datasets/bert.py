import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir))
sys.path.append(root_path)

import torch
from utilities.vqa.dataset import *
from transformers import BertTokenizer
from datasets.creator import DatasetCreator, MultiPurposeDataset
from torch.utils.data import Dataset
from utilities.evaluation.beam_search import BeamSearchInput
from collections import Counter

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def create_datasets(base_path):
    seq_counter = {
        'training': Counter(),
        'testing': Counter()
    }

    def elem_processing_fn(question_id, question, image_path, answer, split):

        question_tkn = bert_tokenizer.encode(question)
        question_tkn = [bert_tokenizer.cls_token_id] + question_tkn + [bert_tokenizer.sep_token_id]
        question_tkn_len = len(question_tkn)

        if split == 'training':

            answer_tkn = bert_tokenizer.encode(answer)
            answer_tkn = answer_tkn + [bert_tokenizer.sep_token_id]
            answer_tkn_len = len(answer_tkn)

            seq_counter[split].update([answer_tkn_len + question_tkn_len])

            sequence = question_tkn + answer_tkn

            input_mask = [1] * (question_tkn_len + answer_tkn_len)
            token_types = [0] * question_tkn_len + [1] * answer_tkn_len

            return [question_id, sequence, token_types, input_mask]
        else:
            seq_counter[split].update([question_tkn_len])
            token_types = [0] * question_tkn_len
            return [question_id, question_tkn, token_types]

    def post_processing_fn(tr_data, ts_data, tr_size, ts_size):

        tr_removed = len(tr_data)

        print('Removing short samples checking frequencies')
        tr_data = list(filter(lambda item: seq_counter['training'][len(item[1])] > 1000, tr_data))

        print(seq_counter)

        tr_removed -= len(tr_data)

        print('Removed {} from training data and {} from testing data'.format(tr_removed, 0))

        tr_data = tr_data[:tr_size]

        print('Len tr = {}, len ts = {}'.format(len(tr_data), len(ts_data)))

        max_len_tr = 0
        max_len_ts = 0

        for length, freq in seq_counter['training'].items():
            if freq > 1000:
                if max_len_tr < length:
                    max_len_tr = length
        for length, freq in seq_counter['testing'].items():
            if max_len_ts < length:
                max_len_ts = length
        # Pad sequences
        print('Padding training sequences..')
        tr_data = DatasetCreator.pad_sequences(tr_data, axis=1, value=int(bert_tokenizer.pad_token_id),
                                               maxlen=max_len_tr)

        tr_data = DatasetCreator.pad_sequences(tr_data, axis=2, value=int(1),
                                               maxlen=max_len_tr)

        tr_data = DatasetCreator.pad_sequences(tr_data, axis=3, value=int(0),
                                               maxlen=max_len_tr)

        return tr_data, ts_data

    DatasetCreator().create_together(tr_size=1000000, ts_size=100000,
                                     tr_destination=os.path.join(base_path, 'training'),
                                     ts_destination=os.path.join(base_path, 'testing'),
                                     elem_processing_fn=elem_processing_fn, post_processing_fn=post_processing_fn)


class BertBeamSearchInput(BeamSearchInput):
    def __init__(self, seq_idx, seg_id, logits_idx, *args):
        super().__init__(seq_idx, logits_idx, *args)
        self.seg_id = seg_id

    def update_args(self, running_args, initial_args):
        """
        We have to update the segment id tensors every time a word is generated in BERT
        """

        running_args[self.seg_id] = torch.cat(
            [running_args[self.seg_id], torch.ones(running_args[self.seg_id].shape[0], 1).long().to('cuda')], dim=1)

        initial_args[self.seg_id] = torch.cat([initial_args[self.seg_id], torch.ones(1).long().to('cuda')])
        return running_args, initial_args


class BertDataset(MultiPurposeDataset):

    def __getitem__(self, item):

        sample = self.data[item]
        if not self.evaluating:
            _, sequence, token_types, input_mask = sample
        else:
            __id, question, token_types = sample

        if not self.evaluating:
            # Return answer + image + length
            return torch.tensor(sequence).long(), torch.tensor(token_types).long(), torch.tensor(input_mask).long()
        else:
            question = torch.tensor(question).long()
            token_types = torch.tensor(token_types).long()
            beam_input = BertBeamSearchInput(0, 1, 0, question, token_types)
            ground_truths = self.evaluation_data[str(__id)]
            return __id, beam_input, ground_truths


if __name__ == '__main__':
    path = resources_path('models', 'baseline', 'answering', 'bert', 'data')
    create_datasets(path)
