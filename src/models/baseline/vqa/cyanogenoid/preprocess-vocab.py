import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir, os.pardir, os.pardir))
sys.path.append(root_path)

from utilities import paths

import json
from collections import Counter
import itertools

import models.baseline.vqa.cyanogenoid.config as config
import models.baseline.vqa.cyanogenoid.data as data
import models.baseline.vqa.cyanogenoid.utils as utils


def extract_vocab(iterable, top_k=None, start=0):
    """ Turns an iterable of list of tokens into a vocabulary.
        These tokens could be single answers or word tokens in questions.
    """
    all_tokens = itertools.chain.from_iterable(iterable)
    counter = Counter(all_tokens)
    if top_k:
        most_common = counter.most_common(top_k)
        most_common = (t for t, c in most_common)
    else:
        most_common = counter.keys()
    # descending in count, then lexicographical order
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens, start=start)}
    return vocab


def main():
    questions = utils.path_for(train=True, question=True)
    answers = utils.path_for(train=True, answer=True)
    # questions = paths.resources_path('data/vqa/Questions/v2_OpenEnded_mscoco_val2014_questions.json')
    # answers = paths.resources_path('data/vqa/Annotations/v2_mscoco_val2014_annotations.json')

    with open(questions, 'r') as fd:
        questions = json.load(fd)
    with open(answers, 'r') as fd:
        answers = json.load(fd)

    questions = data.prepare_questions(questions)
    answers = data.prepare_answers(answers)

    """
    vggpt2_pred_path = paths.resources_path('100K_predictions', 'beam_size_1', 'maxlen_20', 'vggpt2.json')

    with open(vggpt2_pred_path, 'r') as fp:
        vggpt2_preds = json.load(fp)

    good_ids = [int(k) for k in list(vggpt2_preds.keys())]

    questions['questions'] = list(filter(lambda question: question['question_id'] in good_ids, questions['questions']))
    assert len(questions['questions']) == len(good_ids)

    answers['annotations'] = list(filter(lambda answer: answer['question_id'] in good_ids, answers['annotations']))
    assert len(answers['annotations']) == len(good_ids)

    questions = data.prepare_questions(questions)
    answers = data.prepare_answers(answers)
    """
    question_vocab = extract_vocab(questions, start=1)
    answer_vocab = extract_vocab(answers, top_k=config.max_answers)

    vocabs = {
        'question': question_vocab,
        'answer': answer_vocab,
    }
    with open(config.vocabulary_path, 'w') as fd:
        json.dump(vocabs, fd)


if __name__ == '__main__':
    main()
