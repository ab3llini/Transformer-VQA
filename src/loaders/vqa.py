import os
from vqaTools.vqa import VQA
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from pytorch_transformers import BertTokenizer
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
import pickle
from torchvision import transforms
from PIL import Image
import collections
import skimage.io as io
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenize = lambda text: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

this_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.abspath(os.path.join(this_path, os.pardir))

dataDir = os.path.join(src_path, '../data/vqa')
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
        self.sequence = None
        self.token_type_ids = None

    def pad_sequence(self, to_size):
        self.sequence = VQADataset.pad_sequence(self.tkn_question, self.tkn_answer, to_size, tokenizer.pad_token_id)

    def compute_token_type_ids(self):
        self.token_type_ids = [0] * (len(self.tkn_question) + 2) + [1] * (
                    len(self.sequence) - (len(self.tkn_question) + 2))

    def __str__(self):
        io.imshow(io.imread(image_path(self.image_id)))
        io.show()
        return 'Question: {}\nAnswer:{}\nTokenized Question: {}\nTokenized Answer: {}\nSequence: {}\n Types: {}\n ' \
               'Image: {}'.format(self.question, self.answer, self.tkn_question, self.tkn_answer, self.sequence,
                                  self.token_type_ids,
                                  self.image_id)

    def __len__(self):
        return len(self.tkn_question) + len(self.tkn_answer) if self.sequence is None else len(self.sequence)

    def get_sample(self):
        return self.sequence, self.token_type_ids, Image.open(image_path(self.image_id))


class VQADataset(Dataset):

    def __init__(self,
                 fname=None,
                 caching=True,
                 rebuild=False,
                 limit=None,
                 ):

        self.samples = []
        self.tokenizer = tokenizer

        if not rebuild:
            print("Looking for cached samples named '{}'".format(fname))
            if not self.load(fname):
                print("Samples not found, rebuilding dataset")
                self.build(limit=limit)
                if caching:
                    print('Dumping dataset to file "{}"..'.format(fname))
                    # If needed, cache the samples. You might want to do it since it takes a lot to build the dataset
                    self.save(fname) if fname is not None else self.save()
            else:
                print("Samples found and loaded in RAM")
        else:
            print("Rebuilding dataset")

            self.build(limit=limit)
            if caching:
                # If needed, cache the samples. You might want to do it since it takes a lot to build the dataset
                self.save(fname) if fname is not None else self.save()

    def build(self, limit):
        # Load VQA helpers and create indexes
        vqa = VQA(annFile, quesFile)

        # Get QA instances
        qa = vqa.loadQA(vqa.getQuesIds())

        # First, build a more helpful structure.
        # While building, keep track of the samples per length
        candidates = []
        lengths = {}
        n_grayscale = 0

        print('Building candidate list and computing lengths')
        for elem in tqdm(qa):
            _id, q, anss, i = self.unpack_element(elem, vqa)
            for a in anss:
                if self.is_rgb(i):
                    sample = Sample(_id, q, a, i)
                    sample_len = len(sample)
                    if sample_len in lengths:
                        lengths[sample_len] += 1
                    else:
                        lengths[sample_len] = 1
                    candidates.append(sample)
                else:
                    n_grayscale += 1

        print('Done, in the process we dropped {} images because they were in grayscale'.format(n_grayscale))

        print('Ordering candidates by length.. This might take a while..')
        candidates = sorted(candidates, key=lambda _candidate: len(_candidate), reverse=True)

        kept, dropped = [], []
        for length, n_candidates in lengths.items():
            if n_candidates >= 100:
                kept.append(length)
            else:
                dropped.append(length)

        print('Dropping candidates whose length is in the following list: {}'.format(dropped))
        selected = list(filter(lambda _candidate: len(_candidate) in kept, candidates))

        longest_candidate = len(selected[0])

        print('Padding sequences..')
        for candidate in tqdm(selected):
            candidate.pad_sequence(to_size=longest_candidate)

        self.samples = selected

    @staticmethod
    def unpack_element(data, vqa):
        q_id = data['question_id']
        # Access actual question rather than it's ID
        question = vqa.qqa[q_id]['question']
        # Pre compute actual image path
        image_id = data['image_id']
        # Extract only answers and get rid of additional annotations
        answers = [q['answer'] for q in data['answers']]

        return q_id, question, answers, image_id

    @staticmethod
    def pad_sequence(tkn_question, tkn_answer, padding_size, padding_token):

        return tkn_question + tkn_answer + (
                [padding_token] * (padding_size - len(tkn_question) - len(tkn_answer))
        )

    def save(self, fname='samples.pk'):
        with open(os.path.join(this_path, fname), 'wb') as fd:
            pickle.dump(self.samples, fd)

    def load(self, fname='samples.pk'):
        try:
            with open(os.path.join(this_path, fname), 'rb') as fd:
                self.samples = pickle.load(fd)
            return True
        except (OSError, IOError) as e:
            return False

    @staticmethod
    def is_rgb(image_id):
        return Image.open(image_path(image_id)).mode == 'RGB'

    @staticmethod
    def transform_image(image):

        if image.mode != 'RGB':
            raise Exception('Grayscale detection')

        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        return transform(image)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = self.samples[item]
        token_ids, token_type_ids, img = sample.get_sample()
        return torch.tensor(token_ids), torch.tensor(token_type_ids), self.transform_image(img)


if __name__ == '__main__':

    dataset = VQADataset(fname='bert_vgg_padded_types_dataset.pk')
    for s in tqdm(dataset.samples):
        s.compute_token_type_ids()

    dataset.save(fname='bert_vgg_padded_types_dataset.pk')
    for sample in dataset.samples[:10]:
        print(sample, '\n' + '*' * 100)
