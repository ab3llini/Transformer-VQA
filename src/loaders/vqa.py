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

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<PAD>')
tokenize = lambda text: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

# Add the <pad> token to the vocabulary
SPECIAL_TOKENS = ["<pad>"]
# tokenizer.set_special_tokens(SPECIAL_TOKENS)

'''
# Set the number of special tokens in the model
model.set_num_special_tokens(len(SPECIAL_TOKENS))

Examples::

        import torch
        from pytorch_transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
        
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2DoubleHeadsModel.from_pretrained('gpt2')
        
        # Add a [CLS] to the vocabulary (we should train it also!)
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size
        print(tokenizer.cls_token_id, len(tokenizer))  # The newly token the last token of the vocabulary
        
        choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
        encoded_choices = [tokenizer.encode(s) for s in choices]
        cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

        input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
        mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

        outputs = model(input_ids, mc_token_ids=mc_token_ids)
        lm_prediction_scores, mc_prediction_scores = outputs[:2]

'''

# Get the <pad> token's index
pad_idx = tokenizer.convert_tokens_to_ids(['<pad>'])[0]

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
                 caching=True,
                 rebuild=False,
                 q_min_len=5,
                 q_max_len=9,
                 a_min_len=1,
                 a_max_len=2,
                 n_answers=3,
                 limit=None
                 ):

        self.samples = []
        self.tokenizer = tokenizer

        if not rebuild:
            print("Looking for cached samples")
            if not self.load():
                print("Samples not found, rebuilding dataset")
                self.build(q_min_len, q_max_len, a_min_len, a_max_len, n_answers, save=caching, limit=limit)
            else:
                print("Samples found and loaded in RAM")
        else:
            print("Rebuilding dataset")
            self.build(q_min_len, q_max_len, a_min_len, a_max_len, n_answers, save=caching, limit=limit)

    def build(self, q_min_len, q_max_len, a_min_len, a_max_len, n_answers, save, limit):

        # Load VQA helpers and create indexes
        vqa = VQA(annFile, quesFile)

        # Get QA instances
        qa = vqa.loadQA(vqa.getQuesIds())

        print('JSON files parsed, initializing filtering procedure.')

        print('Keeping questions in range {}, answers in range {}, questions with at least {} answers'
              .format(range(q_min_len, q_max_len + 1), range(a_min_len, a_max_len + 1), n_answers))

        sample_size = 0
        grayscale_dropped = 0

        # Create meaningful structure.. VQA Helpers are really bad!
        for _qa in tqdm(qa):
            q_id = _qa['question_id']
            # Access actual question rather than it's ID
            question = vqa.qqa[q_id]['question']
            # Pre compute actual image path
            image_path = imgDir + 'COCO_' + dataSubType + '_' + str(_qa['image_id']).zfill(12) + '.jpg'
            # Extract only answers and get rid of additional annotations
            answers = [q['answer'] for q in _qa['answers']]

            try:
                # Tokenize and filter
                tkn_question, tkn_answers = self.filter(question, answers, image_path, q_min_len, q_max_len, a_min_len,
                                                        a_max_len,
                                                        n_answers)

                # Create and pad sequences
                padded_sequences = self.create_and_pad_sequences(tkn_question, tkn_answers, q_max_len + a_max_len)

                # Add these sequences to our samples, including question id & image path
                self.samples += self.create_samples(q_id, padded_sequences, image_path)

                # Keep track of how many samples we have created
                sample_size += n_answers

                # Print checkpoint
                if sample_size % (n_answers * 100000) == 0:
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

        # If needed, cache the samples. You might want to do it since it takes a lot to build the dataset
        if save:
            self.save()

    # This method creates samples for the given padded sequences including all the information required
    @staticmethod
    def create_samples(q_id, padded_seqs, image_path):
        new = []
        for seq in padded_seqs:
            new.append({'id': q_id, 'seq': seq, 'img': image_path})
        return new

    # This method filters and tokenizes the question/answers
    @staticmethod
    def filter(question, answers, image, q_min_len, q_max_len, a_min_len, a_max_len, n_answers):

        # First and foremost, drop all grayscale images and relative questions.
        # Sadly VGG can't deal with these images properly
        if Image.open(image).mode != 'RGB':
            raise Exception('Grayscale image')

        tokenized_question = tokenize(question)
        tokenized_answers = []

        if len(tokenized_question) not in range(q_min_len, q_max_len + 1):
            raise Exception('Too long or short')

        for answer in answers:
            tokenized_answer = tokenize(answer)
            if len(tokenized_answer) not in range(a_min_len, a_max_len + 1):
                continue
            else:
                tokenized_answers.append(tokenized_answer)

            # Stop if we reach the desired amount of answer for this question
            if len(tokenized_answers) == n_answers:
                break

        # Check consistency
        if len(tokenized_answers) < n_answers:
            raise Exception('Not enough answers')

        return tokenized_question, tokenized_answers

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

    def save(self):
        with open(os.path.join(this_path, 'samples.pk'), 'wb') as fd:
            pickle.dump(self.samples, fd)

    def load(self):
        try:
            with open(os.path.join(this_path, 'samples.pk'), 'rb') as fd:
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
