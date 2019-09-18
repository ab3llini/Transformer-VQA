import os
from vqaTools.vqa import VQA
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from pytorch_transformers import GPT2Tokenizer
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
import pickle
from torchvision import transforms
from PIL import Image
import collections

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<PAD>')
tokenize = lambda text: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

# Get the <pad> token's index
pad_idx = tokenizer.convert_tokens_to_ids(['<PAD>'])[0]

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


# High level representation of a sample
class PrintableSample:
    def __init__(self, _id, tokenized_sequences, image_path):
        self._id = _id
        self.seqs = tokenized_sequences
        self.image_path = image_path

    def plot(self):
        img = self.load_image()
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def load_image(self):
        return Image.open(self.image_path)

    def __str__(self):
        self.plot()
        return 'ID: {}\nSequences {}\nImage {}'.format(
            self._id,
            [tokenizer.decode(seq) for seq in self.seqs],
            self.image_path
        )


class VQADataset(Dataset):

    def __init__(self,
                 fname=None,
                 caching=True,
                 rebuild=False,
                 q_min_len=5,
                 q_max_len=9,
                 a_min_len=1,
                 a_max_len=2,
                 n_answers=3,
                 limit=None,
                 filter_method='longest'
                 ):

        self.samples = []
        self.tokenizer = tokenizer

        if not rebuild:
            print("Looking for cached samples named '{}'".format(fname))
            if not self.load(fname):
                print("Samples not found, rebuilding dataset")
                if filter_method == 'range':
                    self.build_in_range(q_min_len, q_max_len, a_min_len, a_max_len, n_answers, limit=limit)
                    # If needed, cache the samples. You might want to do it since it takes a lot to build the dataset
                    self.save(fname) if fname is not None else self.save()
                else:
                    self.build(limit=limit)
                    # If needed, cache the samples. You might want to do it since it takes a lot to build the dataset
                    self.save(fname) if fname is not None else self.save()
            else:
                print("Samples found and loaded in RAM")
        else:
            print("Rebuilding dataset")
            if filter_method == 'range':
                self.build_in_range(q_min_len, q_max_len, a_min_len, a_max_len, n_answers, limit=limit)
                # If needed, cache the samples. You might want to do it since it takes a lot to build the dataset
                self.save(fname) if fname is not None else self.save()
            else:
                self.build(limit=limit)
                # If needed, cache the samples. You might want to do it since it takes a lot to build the dataset
                self.save(fname) if fname is not None else self.save()

    def build(self, limit):
        # Load VQA helpers and create indexes
        vqa = VQA(annFile, quesFile)

        # Get QA instances
        qa = vqa.loadQA(vqa.getQuesIds())

        answers_size = {}

        print('Computing answer length structure')
        # Create meaningful structure.. VQA Helpers are really bad!
        for data in tqdm(qa):
            # Parse the question
            q_id, question, answers, image_path = self.parse_question(data, vqa)

            for answer in answers:
                tokenized_answer = tokenize(answer)
                _len = len(tokenized_answer)
                if _len in answers_size:
                    answers_size[_len].append({'q_id': q_id, 'q': question, 'a': answer, 'i': image_path})
                else:
                    answers_size[_len] = [{'q_id': q_id, 'q': question, 'a': answer, 'i': image_path}]

        print('Sorting by question length')
        for a_len, data in answers_size.items():
            print('Considering answers {} token long, found {} samples'.format(a_len, len(data)))
            print('Sorting by question length..')
            answers_size[a_len] = sorted(data, key=lambda k: len(k['q']))

        print('Building dataset..')
        answers_size = collections.OrderedDict(sorted(answers_size.items()))

        added = 0

        longest = [key for key, value in answers_size][0]
        input_size = len(longest['q']) + len(longest['a'])

        print('The input size is {} due to the sample : {}'.format(input_size, longest))

        for _len, data in answers_size.items():
            for sample in data:
                if added <= limit:
                    tk_q = tokenize(sample['q'])
                    tk_a = tokenize(sample['a'])
                    pad_qa = self.create_and_pad_sequences(tk_q, tk_a, input_size)
                    self.samples += self.create_samples(sample['q_id'], tk_q, tk_a, pad_qa, sample['i'])

        print('Dataset created, here are some examples')

        for e in self.samples[:10]:
            print(e)

        print('Saving..')

    def build_in_range(self, q_min_len, q_max_len, a_min_len, a_max_len, n_answers, limit):

        # Load VQA helpers and create indexes
        vqa = VQA(annFile, quesFile)

        # Get QA instances
        qa = vqa.loadQA(vqa.getQuesIds())

        sample_size = 0
        grayscale_dropped = 0

        # Create meaningful structure.. VQA Helpers are really bad!
        for data in tqdm(qa):
            # Parse the question
            q_id, question, answers, image_path = self.parse_question(data=data, vqa=vqa)

            try:
                # Tokenize and filter
                tkn_question, tkn_answers = self.filter_in_range(question, answers, image_path, q_min_len, q_max_len, a_min_len,
                                                                 a_max_len,
                                                                 n_answers)

                # Create and pad sequences
                padded_sequences = self.create_and_pad_sequences(tkn_question, tkn_answers, q_max_len + a_max_len)

                # Add these sequences to our samples, including question id & image path
                self.samples += self.create_samples(q_id, tkn_question * len(padded_sequences), tkn_answers, padded_sequences, image_path)

                # Keep track of how many samples we have created
                sample_size += n_answers

                # Print checkpoint
                if sample_size % (n_answers * 10000) == 0:
                    print('Checkpoint : created {} samples'.format(sample_size))

                # Check if we need to stop
                if limit is not None and sample_size > limit:
                    # Make sure to have exactly 'limit' samples
                    self.samples = self.samples[:limit]
                    # Break procedure
                    break
            except Exception as e:
                if str(e) == 'Grayscale image':
                    grayscale_dropped += 1

        print('We had to drop {} samples because they had grayscale images :('.format(grayscale_dropped))

    @staticmethod
    def parse_question(data, vqa):
        q_id = data['question_id']
        # Access actual question rather than it's ID
        question = vqa.qqa[q_id]['question']
        # Pre compute actual image path
        image_path = imgDir + 'COCO_' + dataSubType + '_' + str(data['image_id']).zfill(12) + '.jpg'
        # Extract only answers and get rid of additional annotations
        answers = [q['answer'] for q in data['answers']]

        return q_id, question, answers, image_path

    # This method creates samples for the given padded sequences including all the information required
    @staticmethod
    def create_samples(q_id, questions, answers, padded_seqs, image_path):
        new = []
        for q, a, seq in zip(questions, answers, padded_seqs):
            new.append({'id': q_id, 'q': q, 'a' : a, 'seq': seq, 'img': image_path})
        return new

    # This method filters and tokenizes the question/answers
    @staticmethod
    def filter_in_range(question, answers, image, q_min_len, q_max_len, a_min_len, a_max_len, n_answers):

        # First and foremost, drop all grayscale images and relative questions.
        # Sadly VGG can't deal with these images properly
        if Image.open(image).mode != 'RGB':
            raise Exception('Grayscale image')

        tokenized_question = tokenize(question)
        tokenized_answers = []

        if q_max_len is not None and q_min_len is not None:
            if len(tokenized_question) not in range(q_min_len, q_max_len + 1):
                raise Exception('Too long or short')

        for answer in answers:
            tokenized_answer = tokenize(answer)
            if a_max_len is not None and a_min_len is not None:
                if len(tokenized_answer) not in range(a_min_len, a_max_len + 1):
                    continue
                else:
                    tokenized_answers.append(tokenized_answer)
            else:
                tokenized_answers.append(tokenized_answer)

            # Stop if we reach the desired amount of answer for this question
            if len(tokenized_answers) == n_answers:
                break

        # Check consistency
        if len(tokenized_answers) < n_answers:
            raise Exception('Not enough answers')

        return tokenized_question, tokenized_answers

    # This method filters the samples by keeping the longest sequences first
    @staticmethod
    def filter_longest():
        return 4

    @staticmethod
    def decode_sample(sample):
        return PrintableSample(sample['id'], sample['seq'], sample['img'])

    @staticmethod
    def create_and_pad_sequences(tkn_question, tkn_answers, block_size):
        padded_seqs = []
        for tkn_answer in tkn_answers:
            seq = tkn_question + tkn_answer + (
                    [tokenizer.pad_token_id] * (block_size - len(tkn_question) - len(tkn_answer))
            )
            padded_seqs.append(seq)
        return padded_seqs

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
    def transform_image(image):

        if image.mode != 'RGB':
            raise Exception('Fukin grayscale!')

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
        return torch.tensor(sample['seq']), self.transform_image(Image.open(sample['img']))


if __name__ == '__main__':
    VQADataset(fname='longest_seqs.pk', limit=500000)