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

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def create_datasets(base_path):
    longest_answer = {'training': 0, 'testing': 0}

    def elem_processing_fn(question_id, question, image_path, answer, split):
        question_tkn = bert_tokenizer.encode(question)
        answer_tkn = bert_tokenizer.encode(answer)
        question_tkn_len = len(question_tkn)
        answer_tkn_len = len(answer_tkn)

        if longest_answer[split] < answer_tkn_len:
            longest_answer[split] = answer_tkn_len

        if longest_answer[split] < answer_tkn_len + question_tkn_len:
            longest_answer[split] = answer_tkn_len + question_tkn_len

        # Add BOS, SEP & EOS tokens
        question_tkn = [bert_tokenizer.cls_token_id] + question_tkn + [bert_tokenizer.sep_token_id]
        answer_tkn = answer_tkn + [bert_tokenizer.sep_token_id]

        # Concatenate Q+A
        sequence = question_tkn + answer_tkn

        # Input mask
        input_mask = [1] * (question_tkn_len + 2 + answer_tkn_len + 1)

        # Position mask
        token_types = [0] * (question_tkn_len + 2) + [1] * (answer_tkn_len + 1)

        return [question_id, sequence, token_types, input_mask, question_tkn_len + 2]

    def post_processing_fn(tr_data, ts_data):

        print('Longest = {}'.format(longest_answer))

        # Pad sequences
        print('Padding training sequences..')
        tr_data = DatasetCreator.pad_sequences(tr_data, axis=1, value=int(bert_tokenizer.pad_token_id),
                                               maxlen=longest_answer['training'] + 3)

        tr_data = DatasetCreator.pad_sequences(tr_data, axis=2, value=int(1),
                                               maxlen=longest_answer['training'] + 3)

        tr_data = DatasetCreator.pad_sequences(tr_data, axis=3, value=int(0),
                                               maxlen=longest_answer['training'] + 3)

        # Pad sequences
        print('Padding testing sequences..')
        ts_data = DatasetCreator.pad_sequences(ts_data, axis=1, value=int(bert_tokenizer.pad_token_id),
                                               maxlen=longest_answer['testing'] + 3)

        ts_data = DatasetCreator.pad_sequences(ts_data, axis=2, value=int(1),
                                               maxlen=longest_answer['testing'] + 3)

        ts_data = DatasetCreator.pad_sequences(ts_data, axis=3, value=int(0),
                                               maxlen=longest_answer['testing'] + 3)

        return tr_data, ts_data

    DatasetCreator().create_together(tr_size=1000000, ts_size=200000,
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
            _, sequence, token_types, input_mask, _ = sample
        else:
            __id, sequence, token_types, input_mask, ql = sample

        if not self.evaluating:
            # Return answer + image + length
            return torch.tensor(sequence).long(), torch.tensor(token_types).long(), torch.tensor(input_mask).long()
        else:
            question = torch.tensor(sequence[:ql]).long()
            token_types = torch.tensor(token_types[:ql]).long()
            beam_input = BertBeamSearchInput(0, 1, 0, question, token_types)
            ground_truths = self.evaluation_data[str(__id)]
            return beam_input, ground_truths


if __name__ == '__main__':
    path = resources_path('models', 'baseline', 'answering', 'bert', 'data')
    create_datasets(path)
