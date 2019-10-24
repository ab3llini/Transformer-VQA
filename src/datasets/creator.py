import sys
import os
from collections import Counter

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir))
sys.path.append(root_path)

from utilities.vqa.dataset import *
import keras.preprocessing as k_preproc
import random
import json
import nltk
from torch.utils.data import Dataset


class DatasetCreator:

    def __init__(self, max_answers_per_question=4):
        self.max_answers_per_question = max_answers_per_question

    @staticmethod
    def __load_cached(split):
        assert split in ['training', 'testing'], Exception('Undefined split in load_cached arg')

        path = data_path('cache', '{}.json'.format(split))
        if os.path.exists(path):
            print('{} cache found'.format(split))
            with open(path, 'r') as fp:
                cache = json.load(fp)
        else:
            print('{} cache not found'.format(split))
            cache = None
        return cache

    def __build(self, split, size):
        assert split in ['training', 'testing'], Exception('Undefined split in build arg')

        print('Building {} cache, size: {}'.format(split, size))
        question_path, annotation_path, image_path = get_data_paths(
            data_type='train' if split == 'training' else 'test')
        vqa_helper = VQA(annotation_path, question_path)
        qa_objects = vqa_helper.loadQA(vqa_helper.getQuesIds())

        l_qa = len(qa_objects)
        max_size = l_qa * self.max_answers_per_question
        assert size <= max_size, Exception(
            'Size ({}) should be at maximum {} ({} * {})'.format(size, max_size, l_qa, self.max_answers_per_question))

        cache = []
        question_answer_map = {}
        sequence_len_counter = Counter()

        # Skip if image is not RGB
        for qa_object in tqdm(qa_objects):

            # Parse object
            obj_id, obj_q, obj_as, obj_i = get_qai(qa_object, vqa_helper)
            q_tkn_len = len(nltk.word_tokenize(obj_q))
            image_full_path = image_path + str(obj_i).zfill(12) + '.jpg'
            if not check_rgb(image_full_path):
                continue
            # Performance booster.
            prev_a = None
            prev_a_tkn = None
            prev_a_tkn_len = None
            # Every question has 10 answers
            for obj_a in obj_as:
                # Try to skip tokenizing if possible and use cached version
                if obj_a != prev_a:
                    # Embed the question and answer
                    a_tkn = nltk.word_tokenize(obj_a)
                    a_tkn_len = len(a_tkn)
                    prev_a = obj_a
                    prev_a_tkn = a_tkn
                    prev_a_tkn_len = a_tkn_len

                if split == 'testing':
                    # Update the question answer map
                    if obj_id not in question_answer_map:
                        question_answer_map[obj_id] = [prev_a_tkn]
                    else:
                        question_answer_map[obj_id].append(prev_a_tkn)

                # Update the sequence len counter
                sequence_len_counter.update([q_tkn_len + prev_a_tkn_len])

                # Add sample to cache
                cache.append([int(obj_id), obj_q, image_full_path, prev_a, q_tkn_len, prev_a_tkn_len])

            # if len(cache) > 10000:
            # break

        print('Removing elements whose sequence length frequency count is under 1000')
        # Now we remove all the samples above the threshold
        # But if we have, say, 50 samples long 20 and 1M samples long 21
        # we keep these 50 samples since we'll have to pad anyway up to 21
        sorted_lengths = sorted(list(sequence_len_counter.keys()), reverse=True)
        bad_lengths = []

        for length in sorted_lengths:
            if sequence_len_counter[length] < 1000:
                # Remove all these samples
                bad_lengths.append(length)
            else:
                break

        print('Will remove elements whose question + answer is in {}'.format(bad_lengths))

        cache = list(filter(lambda o: o[-2] + o[-1] not in bad_lengths, cache))
        # Switch to numpy
        cache = np.array(cache, dtype=np.object)
        cache = cache.astype(object)
        cache[:, 0] = cache[:, 0].astype(np.int)

        data = []

        question_counter = Counter()

        # Build a dataset keeping first the sequences with the longest answers
        # At the same time make sure not to have more than
        # First we sort the cache in descending order looking at the answer length
        cache = cache[(-cache[:, -1]).argsort()]
        # The we scan it sequentially and build the dataset
        for idx, elem in enumerate(tqdm(cache)):
            if len(data) >= size:
                if question_counter[elem[0]] == 0:
                    if split == 'testing':
                        if elem[0] in question_answer_map:
                            del question_answer_map[elem[0]]
            else:
                if question_counter[elem[0]] < self.max_answers_per_question:
                    data.append(elem[:-2].tolist())  # Remove lengths
                    question_counter.update([elem[0]])

        assert len(data) == size, Exception('Could not build a dataset of the specified size')

        # Dump data
        path = data_path('cache', '{}.json'.format(split))
        evaluation_path = data_path('cache', 'evaluation.json')
        with open(path, 'w+') as fp:
            json.dump(data, fp)
        if split == 'testing':
            with open(evaluation_path, 'w+') as fp:
                json.dump(question_answer_map, fp)

        return data

    @staticmethod
    def pad_sequences(data, axis, value, maxlen):
        if not isinstance(data, (np.ndarray, np.generic)):
            data = np.array(data)
        padded = k_preproc.sequence.pad_sequences(data[:, axis], padding='post',
                                                  value=value, maxlen=maxlen)
        for sample, pad in zip(data, padded):
            sample[axis] = pad.tolist()

        return data.tolist()

    def create(self, split, size, destination, pre_processing_fn=None, elem_processing_fn=None,
               post_processing_fn=None, process_mode='list'):
        assert split in ['training', 'testing'], Exception('Undefined split in create args')
        assert process_mode in ['list', 'dict'], Exception('Undefined elem_process_mode in create args')

        cache, map_cache = self.__load_cached(split=split)
        if cache is None or map_cache is None or len(cache) != size:
            self.__build(split=split, size=size)
            cache, map_cache = self.__load_cached(split=split)

        if destination is None:
            print('No destination was provided.')
            return

        data = cache if process_mode == 'list' else {}

        # Initialize a dictionary that will later be used to dump the datasets using this split
        if pre_processing_fn is not None:
            print('Pre processing..')
            data = pre_processing_fn(data, map_cache) \
                if process_mode == 'list' \
                else pre_processing_fn(cache, map_cache)

        if elem_processing_fn is not None:
            print('Element processing..')
            if pre_processing_fn is None:
                target = cache
            else:
                target = data
            for idx, element in enumerate(tqdm(target)):
                # Iterate over each element in the cache and call all the manipulators sequentially
                question_id, question, image_path, answer = element
                answers = map_cache[str(question_id)]
                if process_mode == 'list':
                    data[idx] = elem_processing_fn(question_id, question, image_path, answer, answers)
                else:
                    key, value = elem_processing_fn(question_id, question, image_path, answer, answers)
                    data[key] = value

        if post_processing_fn is not None:
            print('Post processing..')
            data = post_processing_fn(data, map_cache)

        with open(os.path.join('{}.json'.format(destination)), 'w+') as fp:
            json.dump(data, fp)

    def create_together(self, tr_size, ts_size, tr_destination, ts_destination, pre_processing_fn=None,
                        elem_processing_fn=None,
                        post_processing_fn=None):

        tr_cache = self.__load_cached(split='training')
        ts_cache = self.__load_cached(split='testing')

        if tr_cache is None or len(tr_cache) != tr_size:
            self.__build(split='training', size=tr_size)
            tr_cache = self.__load_cached(split='training')

        if ts_cache is None or len(ts_cache) != ts_size:
            self.__build(split='testing', size=ts_size)
            ts_cache = self.__load_cached(split='testing')

        if tr_destination is None:
            print('No training destination was provided.')
            return

        if ts_destination is None:
            print('No testing destination was provided.')
            return

        tr_data = tr_cache
        ts_data = ts_cache

        # Initialize a dictionary that will later be used to dump the datasets using this split
        if pre_processing_fn is not None:
            print('Pre processing..')
            tr_data, ts_data = pre_processing_fn(tr_data, ts_data)

        if elem_processing_fn is not None:
            print('Element processing..')
            for data, split in [[tr_data, 'training'], [ts_data, 'testing']]:
                for idx, element in enumerate(tqdm(data)):
                    # Iterate over each element in the cache and call all the manipulators sequentially
                    question_id, question, image_path, answer = element
                    data[idx] = elem_processing_fn(question_id, question, image_path, answer, split)

        if post_processing_fn is not None:
            print('Post processing..')
            tr_data, ts_data = post_processing_fn(tr_data, ts_data)

        with open(os.path.join('{}.json'.format(tr_destination)), 'w+') as fp:
            json.dump(tr_data, fp)
        with open(os.path.join('{}.json'.format(ts_destination)), 'w+') as fp:
            json.dump(ts_data, fp)


class MultiPurposeDataset(Dataset):
    def __init__(self, location, split='training', maxlen=None, evaluating=False):
        assert split in ['training', 'testing']
        try:
            with open(os.path.join(location, '{}.json'.format(split)), 'r') as fd:
                self.data = json.load(fd)

            self.maxlen = maxlen if maxlen is not None else len(self.data)
            self.split = split
            self.evaluating = evaluating

            if evaluating:
                evaluation_data_file = resources_path(data_path('cache'), 'evaluation.json')
                with open(evaluation_data_file, 'r') as j:
                    self.evaluation_data = json.load(j)

            print('Data loaded successfully.')

        except (OSError, IOError) as e:
            print('Unable to load data. Did you build it first?', str(e))

    def __len__(self):
        return self.maxlen

    @staticmethod
    def collate_fn(batch):
        beam_inputs = [item[0] for item in batch]
        ground_truths = [item[1] for item in batch]
        return beam_inputs, ground_truths


if __name__ == '__main__':
    # Build only the cache
    DatasetCreator().create(split='training', size=1000000, destination=None)
    DatasetCreator().create(split='testing', size=200000, destination=None)
