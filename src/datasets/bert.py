import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir))
sys.path.append(root_path)

import torch
from utilities.vqa.dataset import *
from transformers import BertTokenizer
from datasets.creator import DatasetCreator
from torch.utils.data import Dataset
from utilities.evaluation.beam_search import BeamSearchInput

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class BertDatasetCreator():
    def __init__(self, tokenizer=None, tr_size=None, ts_size=None, generation_seed=None):
        super().__init__(tr_size, ts_size, generation_seed)

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def embed_fn(self, text):
        """
        Embeds a text sequence using GPT2 tokenizer
        :param text: text to be embedded
        :return: embedded sequence + length
        """
        tkn = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        return tkn, len(tkn)

    def process(self, candidates_tr, candidates_ts):
        # Add tokens to separate Q & A

        longest_tr, longest_ts = [0], [0]

        print('Processing..')
        for split, (candidates, longest) in enumerate([[candidates_tr, longest_tr], [candidates_ts, longest_ts]]):
            for i, sample in tqdm(enumerate(candidates)):
                if longest[0] < sample[self.tkn_a_len_idx] + sample[self.tkn_q_len_idx]:
                    longest[0] = sample[self.tkn_a_len_idx] + sample[self.tkn_q_len_idx]
                # Save some information
                l_q = sample[self.tkn_q_len_idx] + 2  # + BOS & SEP
                l_a = sample[self.tkn_a_len_idx] + 1  # + EOS

                # Add BOS, SEP & EOS tokens
                sample[self.tkn_q_idx] = [self.tokenizer.cls_token_id] + sample[self.tkn_q_idx] + [
                    self.tokenizer.sep_token_id]
                sample[self.tkn_a_idx] = sample[self.tkn_a_idx] + [self.tokenizer.sep_token_id]
                # Concatenate Q+A
                sample[self.tkn_q_idx] += sample[self.tkn_a_idx]
                # Compute sequence length
                seq_len = len(sample[self.tkn_q_idx])
                # If training set save only the sequence length
                # If test set save even question and answer lengths
                if split == 0:
                    sample[self.tkn_q_len_idx] = seq_len
                else:
                    sample[self.tkn_q_len_idx] = [seq_len, l_q, l_a]
                    # Replace answer slots with type ids & mask
                sample[self.tkn_a_idx] = ([0] * l_q + [1] * l_a)  # Replacing answer with token type ids
                sample[self.tkn_a_len_idx] = [1] * (l_q + l_a)  # Replacing answer len with pad mask

            # Account for start, sep & stop tokens added!!
            longest[0] += 3

        if self.size_tr != -1:
            candidates_tr = self.pad_sequences(candidates_tr, axis=1, value=int(self.tokenizer.pad_token_id),
                                               maxlen=longest_tr[0])
            candidates_tr = self.pad_sequences(candidates_tr, axis=3, value=1, maxlen=longest_tr[0])
            candidates_tr = self.pad_sequences(candidates_tr, axis=4, value=0, maxlen=longest_tr[0])

        if self.size_ts != -1:
            candidates_ts = self.pad_sequences(candidates_ts, axis=1, value=int(self.tokenizer.pad_token_id),
                                               maxlen=longest_ts[0])

            candidates_ts = self.pad_sequences(candidates_ts, axis=3, value=1, maxlen=longest_ts[0])

            candidates_ts = self.pad_sequences(candidates_ts, axis=4, value=0, maxlen=longest_ts[0])

        return candidates_tr, candidates_ts


class BertBeamSearchInput(BeamSearchInput):
    def __init__(self, model, seq_idx, seg_id, logits_idx, *args):
        super().__init__(model, seq_idx, logits_idx, *args)
        self.seg_id = seg_id

    def get_logits(self, running_args, args):
        """
        We have to update the segment id tensors every time a word is generated in BERT
        """
        out = self.model(*running_args)
        running_args[self.seg_id] = torch.cat(
            [running_args[self.seg_id], torch.ones(running_args[self.seg_id].shape[0], 1).long().to('cuda')], dim=1)

        args[self.seg_id] = torch.cat([args[self.seg_id], torch.ones(1).long().to('cuda')])
        return out[self.logits_idx], out


class BertDataset():
    """
    This is a dataset specifically crafted for BERT models
    """

    def __init__(self, directory, name, maxlen=None, split='train', bleu_batch=False):
        try:
            with open(os.path.join(directory, name), 'rb') as fd:
                self.data = pickle.load(fd)
            # Get image path
            _, _, self.i_path = get_data_paths(data_type=split)
            self.maxlen = maxlen if maxlen is not None else len(self.data)
            self.bleu_batch = bleu_batch
            print('Data loaded successfully.')
            for e in range(len(self.data) - 1):
                assert self.data[e].shape == self.data[e + 1].shape

            print('Sanity check passed.')
        except (OSError, IOError) as e:
            print('Unable to load data. Did you build it first?', str(e))

    def get_image(self, image_id):
        return load_image(self.i_path, image_id)

    def __getitem__(self, item):
        sample = self.data[item]
        identifier = sample[0]
        sequence = torch.tensor(sample[1]).long()
        token_types = torch.tensor(sample[3]).long()
        att_mask = torch.tensor(sample[4]).long()
        if not self.bleu_batch:
            length = torch.tensor(sample[2]).long()
            return identifier, sequence, token_types, att_mask, length
        else:
            lengths = sample[2]
            seq_len, q_len, a_len = lengths
            seq_len = torch.tensor(seq_len).long()
            q_len = torch.tensor(q_len).long()
            a_len = torch.tensor(a_len).long()
            return identifier, sequence, token_types, att_mask, seq_len, q_len, a_len

    def __len__(self):
        return self.maxlen

    def get_bleu_inputs(self, model, batch, device):
        answer_start_idx = batch[4][0] - batch[6][0]
        question = batch[1][0][:answer_start_idx].to(device)
        token_types = batch[2][0][:answer_start_idx].to(device)
        beam_search_input = BertBeamSearchInput(model, 0, 1, 0, question, token_types)
        ground_truths = []
        for seq, seq_len, a_len in zip(batch[1], batch[4], batch[6]):
            answer_start_idx = seq_len - a_len
            ground_truths.append(seq[answer_start_idx:answer_start_idx + a_len - 1].tolist())

        return beam_search_input, ground_truths


def create(tr_size=1000000, ts_size=100000):
    destination = resources_path('models', 'baseline', 'answering', 'bert', 'data')
    dsc = BertDatasetCreator(tokenizer=bert_tokenizer, tr_size=tr_size, ts_size=ts_size, generation_seed=555)
    dsc.create(destination)


if __name__ == '__main__':
    create()
