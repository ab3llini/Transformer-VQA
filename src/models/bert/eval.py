import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)
sys.path.append('/home/alberto/PycharmProjects/BlindLess/data/vqa')

print(root_path)

import torch
import random
from models.bert import model
from pytorch_transformers import BertTokenizer
from loaders.vqa import VQADataset
from PIL import Image
import collections
import skimage.io as io

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenize = lambda text: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

this_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.abspath(os.path.join(this_path, os.pardir))

dataDir = os.path.join(src_path, '../../data/vqa')
versionType = 'v2_'  # this should be '' when using VQA v2.0 dataset
taskType = 'OpenEnded'  # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType = 'mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType = 'train2014'
annFile = '%s/Annotations/%s%s_%s_annotations.json' % (dataDir, versionType, dataType, dataSubType)
quesFile = '%s/Questions/%s%s_%s_%s_questions.json' % (dataDir, versionType, taskType, dataType, dataSubType)
imgDir = '%s/Images/%s/' % (dataDir, dataSubType)

image_path = lambda image_id: imgDir + 'COCO_' + dataSubType + '_' + str(image_id).zfill(12) + '.jpg'


# High level representation of a sample
class Sample:
    def __init__(self, _id, question, answer, image_id):
        self._id = _id
        self.question = question
        self.answer = answer
        self.image_id = image_id
        self.tkn_question = [tokenizer.cls_token_id] + tokenize(question) + [tokenizer.sep_token_id]
        self.tkn_answer = tokenize(answer) + [tokenizer.sep_token_id]
        self.sequence = None  # The input to the model
        self.masked_sequence = None
        self.token_type_ids = None  # 0 are the question, 1 the answer
        self.attention_mask = None  # 1 are NOT padding tokens, 0 are padding tokens

    def mask_sequence(self):
        start_mask = len(self.tkn_question)
        end_mask = start_mask + len(self.tkn_answer) - 1  # -1 do not account for last sep
        self.masked_sequence = self.sequence.copy()
        self.masked_sequence[start_mask:end_mask] = [tokenizer.mask_token_id] * (end_mask - start_mask)

        assert len(self.masked_sequence) == len(self.sequence)

    def get_pad_start_index(self):
        return 1 + len(self.tkn_question) + 1 + len(self.tkn_answer) + 1

    def pad_sequence(self, to_size):
        padded, mask = VQADataset.pad_sequence(self.tkn_question, self.tkn_answer, to_size, tokenizer.pad_token_id)
        self.sequence = padded
        self.attention_mask = mask

    def compute_token_type_ids(self):
        self.token_type_ids = [0] * (len(self.tkn_question) + 2) + [1] * (
                len(self.sequence) - (len(self.tkn_question) + 2))

    def __str__(self):
        io.imshow(io.imread(image_path(self.image_id)))
        io.show()
        return 'Question: {}' \
               '\nAnswer:{}' \
               '\nTokenized Question: {}' \
               '\nTokenized Answer: {}' \
               '\nSequence: {}' \
               '\nMasked sequence: {}' \
               '\nPadding mask: {}' \
               '\nTypes: {}\n ' \
               'Image: {}'.format(self.question,
                                  self.answer,
                                  self.tkn_question,
                                  self.tkn_answer,
                                  self.sequence,
                                  self.masked_sequence,
                                  self.attention_mask,
                                  self.token_type_ids,
                                  self.image_id)

    def __len__(self):
        return len(self.tkn_question) + len(self.tkn_answer) if self.sequence is None else len(self.sequence)

    def get_sample(self):
        return self.masked_sequence, \
               self.sequence, \
               self.token_type_ids, \
               self.attention_mask, \
               Image.open(image_path(self.image_id))



# Load the model
model = torch.load('bert_vgg_DS%_0.5_B_64_LR_5e-05_CHKP_EPOCH_3.h5')


# Load some data (seed to replicate experiments)
seed = random.seed(867)
dataset = VQADataset(fname='bert_vgg_padded_types_dataset.pk')
idx = random.randint(0, len(dataset))

print(dataset.samples[idx])

# Hardware
device = torch.cuda.current_device()
model.to(device)


image = dataset[idx][4].unsqueeze(0).to(device)
answer = dataset.samples[idx].answer

while True:

    question = input('Question: ')
    question = '[CLS] ' + question + ' [SEP]'
    _input = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(question))
    input_type_ids = [1] * len(_input)
    att_mask = [1] * len(_input)

    next = 0
    limit = 15
    c = 0
    # Evaluate the model
    with torch.no_grad():
        while next != tokenizer.sep_token_id and c < limit:
            out = model(torch.tensor(_input).unsqueeze(0).to(device), torch.tensor(input_type_ids).unsqueeze(0).to(device), torch.tensor(att_mask).unsqueeze(0).to(device), image)

            preds = torch.argmax(out[0], dim=1).tolist()
            next = preds[-1]
            _input += [next]
            input_type_ids += [0]
            att_mask += [1]
            c += 1

    print(tokenizer.decode(_input))