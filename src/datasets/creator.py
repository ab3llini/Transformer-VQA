import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

from utilities.vqa.dataset import *
from torch.utils.data import Dataset


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

    def __create_candidates(self):
        """
        This method iterates over each single element in the dataset.
        :return: A list of candidates in the form:
        [id, tkn question, tkn question len, image id, tkn answer, tkn answer len]
        """

        candidates_tr, candidates_ts = [], []

        # We are working on Open Domain Answering systems
        # Hence we are interested in the longest answers (annotations) first
        objects = [[self.qa_objects_tr, candidates_tr, self.vqa_helper_tr],
                   [self.qa_objects_ts, candidates_ts, self.vqa_helper_ts]]

        print('Building auxiliary candidate structures..')
        for qa_objects, candidates, vqa_helper in objects:
            for qa_object in tqdm(qa_objects):
                # Parse object
                obj_id, obj_q, obj_as, obj_i = get_qai(qa_object, vqa_helper)

                # Embed the question
                q_embed, q_embed_len = self.__embed_fn(obj_q)

                # Performance booster.
                prev_a = None
                prev_a_emb = None

                # Every question has 10 answers
                for obj_a in obj_as:

                    # Try to skip embedding if possible and use cached version
                    if obj_a == prev_a:
                        a_embed = prev_a_emb
                    else:
                        # Embed the question and answer
                        a_embed, a_embed_len = self.__embed_fn(obj_a)
                        prev_a = obj_a
                        prev_a_emb = a_embed

                        # Add sample
                        if a_embed_len not in candidates:
                            candidates[a_embed_len] = [[obj_id, q_embed, q_embed_len, obj_i, a_embed, a_embed_len]]
                        else:
                            candidates[a_embed_len].append([obj_id, q_embed, q_embed_len, obj_i, a_embed, a_embed_len])

        return np.array(candidates_tr), np.array(candidates_ts)

    def __select_candidate_modalities(self, ):
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

        candidates_tr, candidates_ts = self.__create_candidates()

        return candidates_tr[:, axes], candidates_ts[:, axes]

    def __filter_candidates(self, candidates_tr, candidates_ts):
        """
        This method filters out candidates according to a default criterion which selects first
        those candidates whose annotation is longer up to the limit size.
        You can, and are advised to, override this method whenever needed for custom filtering.
        Just make sure to return the data in the correct way
        :return: A tuple of candidates (tr & ts) whose size must match the specified one in the init method
        """

        # Sanity check
        assert self.size_tr <= len(candidates_tr)
        assert self.size_ts <= len(candidates_ts)

        filtered_tr, filtered_ts = [], []

        # Get the indices of the K candidates whose answer length is longest
        # K = size (either tr or ts)
        for candidates, filtered, size in [[candidates_tr, filtered_tr, self.size_tr],
                                           [candidates_ts, filtered_ts, self.size_ts]]:
            # Note: candidates[: -1] returns the length of the tokenized answer at that row
            longest_answers_indices = np.argpartition(candidates[: -1], -size)[-size:]
            # Select only the longest answers
            filtered[:] = candidates[longest_answers_indices]

        return filtered_tr, filtered_ts

    def __embed_fn(self, text):
        """
        Function that tokenizes text. Must return a tuple (tokenized text, len)
        This function is task specific and MUST be overwritten
        :param text: text to be embedded
        :return: (embedded text, embedded text length)
        """
        return None, None

    def create(self):
        """
        Builds the candidates list, selects the modalities and applies basic filtering.
        :return:
        """
        candidates = self.__select_candidate_modalities()
        return self.__filter_candidates(*candidates)

    def __save(self, candidates, location):
        print('Saving dataset to '.format(directory, name))
        with open(os.path.join(directory, 'tr_' + name), 'wb') as fd:
            pickle.dump(selected_tr, fd)

        # Dump dataset
        print('Saving testing dataset to {}/ts_{}'.format(directory, name))
        with open(os.path.join(directory, 'ts_' + name), 'wb') as fd:
            pickle.dump(selected_ts, fd)


class VADatasetCreator(DatasetCreator):
    def __init__(self, tr_size=None, ts_size=None, generation_seed=None):
        super().__init__(tr_size, ts_size, generation_seed, visual=True, question=False)


class QADatasetCreator(DatasetCreator):
    def __init__(self, tr_size=None, ts_size=None, generation_seed=None):
        super().__init__(tr_size, ts_size, generation_seed, visual=False, question=True)


class VQADatasetCreator(DatasetCreator):
    def __init__(self, tr_size=None, ts_size=None, generation_seed=None):
        super().__init__(tr_size, ts_size, generation_seed, visual=True, question=True)