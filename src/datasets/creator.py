import sys
import os
from collections import Counter

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

from utilities.vqa.dataset import *
import keras.preprocessing as k_preproc
import random


class DatasetCreator:
    def __init__(self, size_tr=None, size_ts=None, generation_seed=None, visual=True, question=True):
        """
        :param size_tr:
        :param size_ts:
        :param generation_seed:
        :param visual:
        :param question:
        """

        # Save split sizes
        self.size_tr = size_tr
        self.size_ts = size_ts

        # Save the generation seed
        self.generation_seed = generation_seed

        # Modalities
        self.visual = visual
        self.question = question

        # Set the seed
        if self.generation_seed is not None:
            random.seed(self.generation_seed)

        # Load in RAM the samples using the VQA helper tools and our path parser.
        self.q_path_tr, self.a_path_tr, self.i_path_tr = get_data_paths(data_type='train')
        self.q_path_ts, self.a_path_ts, self.i_path_ts = get_data_paths(data_type='test')

        # Create the helpers
        self.vqa_helper_tr = VQA(self.a_path_tr, self.q_path_tr)
        self.vqa_helper_ts = VQA(self.a_path_ts, self.q_path_ts)

        # Get all the question/answer objects
        self.qa_objects_tr = self.vqa_helper_tr.loadQA(self.vqa_helper_tr.getQuesIds())
        self.qa_objects_ts = self.vqa_helper_ts.loadQA(self.vqa_helper_ts.getQuesIds())

    def create_candidates(self):
        """
        This method iterates over each single element in the dataset.
        :return: A list of candidates in the form:
        [id, tkn question, tkn question len, image id, tkn answer, tkn answer len]
        """

        candidates_tr, candidates_ts = [], []
        not_rgb = 0

        # We are working on Open Domain Answering systems
        # Hence we are interested in the longest answers (annotations) first
        objects = [[self.qa_objects_tr, candidates_tr, self.vqa_helper_tr, self.i_path_tr],
                   [self.qa_objects_ts, candidates_ts, self.vqa_helper_ts, self.i_path_ts]]
        print('Building auxiliary candidate structures..')
        for qa_objects, candidates, vqa_helper, image_paths in objects:
            # c = 0
            # Skip if image is not RGB
            for qa_object in tqdm(qa_objects):

                # Parse object
                obj_id, obj_q, obj_as, obj_i = get_qai(qa_object, vqa_helper)
                if not check_rgb(image_paths, obj_i):
                    not_rgb += 10
                    continue

                # Embed the question
                q_embed, q_embed_len = self.embed_fn(obj_q)

                # Performance booster.
                prev_a = None
                prev_a_emb = None
                prev_a_emb_len = None
                # Every question has 10 answers
                for obj_a in obj_as:
                    # Try to skip embedding if possible and use cached version
                    if obj_a != prev_a:
                        # Embed the question and answer
                        a_embed, a_embed_len = self.embed_fn(obj_a)
                        prev_a = obj_a
                        prev_a_emb = a_embed
                        prev_a_emb_len = a_embed_len

                    # Add sample
                    candidates.append([obj_id, q_embed, q_embed_len, obj_i, prev_a_emb, prev_a_emb_len])
                # if c > 100:
                #     break
                # else:
                #     c += 1
        print('Non RGB samples (removed) = {}'.format(not_rgb))

        candidates_tr, candidates_ts = np.array(candidates_tr), np.array(candidates_ts)

        # Preliminary filtering operation. This MUST be done here to ensure
        # equal samples across different types of data sets with same seed
        # Minimum threshold
        min_candidates_per_len = 1000

        for split, candidates in enumerate([candidates_tr, candidates_ts]):
            len_counter = Counter()
            len_positions = {}
            # First scan to init filtering counters
            for idx, sample in enumerate(candidates):
                tot_len = sample[2] + sample[5]
                len_counter[tot_len] += 1
                if tot_len not in len_positions:
                    len_positions[tot_len] = [idx]
                else:
                    len_positions[tot_len].append(idx)

            # Now we remove all the samples above the threshold
            # But if we have, say, 50 samples long 20 and 1M samples long 21
            # we keep these 50 samples since we'll have to pad anyway up to 21
            sorted_lengths = sorted(list(len_counter.keys()), reverse=True)
            to_remove = []

            for length in sorted_lengths:
                if len_counter[length] < min_candidates_per_len:
                    # Remove all these samples
                    to_remove.append(length)
                else:
                    break

            remove_indices = []
            for length in to_remove:
                remove_indices += len_positions[length]

            print('Will remove all samples which Q+A len in {} ({} indices)'.format(to_remove, len(remove_indices)))
            if split == 0:
                candidates_tr = np.delete(candidates, obj=remove_indices, axis=0)
            else:
                candidates_ts = np.delete(candidates, obj=remove_indices, axis=0)

        return np.array(candidates_tr), np.array(candidates_ts)

    def select_candidate_modalities(self):
        """
        Selects the desired modalities (Visual, Question, Answer) across all candidates
        :return: A list of candidates with the requested modalities
        """

        if self.visual and self.question:
            axes = [0, 1, 2, 3, 4, 5]  # VQA
        elif self.visual and not self.question:
            axes = [0, 3, 4, 5]  # VA
        elif self.question and not self.visual:
            axes = [0, 1, 2, 4, 5]  # QA
        else:
            axes = [0, 4, 5]  # A

        candidates_tr, candidates_ts = self.create_candidates()

        return candidates_tr[:, axes], candidates_ts[:, axes]

    def filter_candidates(self, candidates_tr, candidates_ts):
        """
        This method filters out candidates according to a default criterion which selects first
        those candidates whose annotation is longer up to the limit size.
        You can, and are advised to, override this method whenever needed for custom filtering.
        Just make sure to return the data in the correct way
        :return: A tuple of candidates (tr & ts) whose size must match the specified one in the init method
        """

        # Sanity check
        if self.size_tr is not None:
            assert self.size_tr <= len(candidates_tr)
        if self.size_ts is not None:
            assert self.size_ts <= len(candidates_ts)

        filtered_tr, filtered_ts = [], []

        # Shuffle samples before selecting them
        np.random.seed(self.generation_seed)

        # Get the indices of the K candidates whose answer length is longest
        # K = size (either tr or ts)
        for candidates, filtered, size in [[candidates_tr, filtered_tr, self.size_tr],
                                           [candidates_ts, filtered_ts, self.size_ts]]:
            np.random.shuffle(candidates)

            if size is None:
                filtered[:] = candidates
                continue
            # Note: candidates[: -1] returns the length of the tokenized answer at that row
            longest_answers_indices = np.argpartition(candidates[:, -1], -size)[-size:]
            # Select only the longest answers
            filtered[:] = candidates[longest_answers_indices]

        return filtered_tr, filtered_ts

    def embed_fn(self, text):
        """
        Function that tokenizes text. Must return a tuple (tokenized text, len)
        This function is task specific and MUST be overwritten
        :param text: text to be embedded
        :return: (embedded text, embedded text length)
        """
        return None, None

    def process(self, candidates_tr, candidates_ts):
        """
        This method is task specific.
        This method is called right after building the dataset to further process candidates
        Overwrite it to achieve custom processing
        :param candidates_tr: unprocessed candidates
        :param candidates_ts: unprocessed candidates
        :return: processed candidates
        """
        return candidates_tr, candidates_ts

    @staticmethod
    def save(candidates, location):
        print('Saving dataset to {}'.format(location))
        with open(location, 'wb') as fd:
            pickle.dump(candidates, fd)

    def build(self):
        """
        Builds the candidates list, selects the modalities and applies basic filtering.
        :return:
        """
        candidates = self.select_candidate_modalities()
        return self.filter_candidates(*candidates)

    @staticmethod
    def pad_sequences(candidates, axis, value, maxlen):
        if not isinstance(candidates, (np.ndarray, np.generic)):
            candidates = np.array(candidates)
        padded = k_preproc.sequence.pad_sequences(candidates[:, axis], padding='post',
                                                  value=value, maxlen=maxlen)
        for sample, pad in zip(candidates, padded):
            sample[axis] = pad

        return candidates

    def create(self, location):
        candidates = self.build()
        set_tr, set_ts = self.process(*candidates)
        self.save(set_tr, os.path.join(location, 'training.pk'))
        self.save(set_ts, os.path.join(location, 'testing.pk'))


class VADatasetCreator(DatasetCreator):
    def __init__(self, tr_size=None, ts_size=None, generation_seed=None):
        super().__init__(tr_size, ts_size, generation_seed, visual=True, question=False)
        self.img_idx = 1
        self.tkn_a_idx = 2
        self.tkn_a_len_idx = 3


class QADatasetCreator(DatasetCreator):
    def __init__(self, tr_size=None, ts_size=None, generation_seed=None):
        super().__init__(tr_size, ts_size, generation_seed, visual=False, question=True)
        self.tkn_q_idx = 1
        self.tkn_q_len_idx = 2
        self.tkn_a_idx = 3
        self.tkn_a_len_idx = 4


class VQADatasetCreator(DatasetCreator):
    def __init__(self, tr_size=None, ts_size=None, generation_seed=None):
        super().__init__(tr_size, ts_size, generation_seed, visual=True, question=True)
        self.tkn_q_idx = 1
        self.tkn_q_len_idx = 2
        self.img_idx = 3
        self.tkn_a_idx = 4
        self.tkn_a_len_idx = 5
