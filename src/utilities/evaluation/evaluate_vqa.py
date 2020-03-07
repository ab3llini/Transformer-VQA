import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

from utilities import paths

import matplotlib.pyplot as plt
from utilities.vqa.dataset import *
from utilities.vqa.vqa import VQA
from utilities.vqa.evaluation import VQAEval


# An example result json file has been provided in './Results' folder.

def vqa_evaluation(questions, annotations, results, output_dir, precision=2):
    # create vqa object and vqaRes object
    vqa = VQA(annotations, questions)
    vqaRes, result_ques_ids = vqa.loadRes(results, questions)

    # create vqaEval object by taking vqa and vqaRes
    vqaEval = VQAEval(vqa, vqaRes,
                      n=precision)  # n is precision of accuracy (number of places after decimal), default is 2

    # evaluate results
    """
    If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
    By default it uses all the question ids in annotation file
    """
    vqaEval.evaluate(quesIds=result_ques_ids)

    # print accuracies
    print("\n")
    print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
    print("Per Question Type Accuracy is the following:")
    for quesType in vqaEval.accuracy['perQuestionType']:
        print("%s : %.02f" % (quesType, vqaEval.accuracy['perQuestionType'][quesType]))
    print("\n")
    print("Per Answer Type Accuracy is the following:")
    for ansType in vqaEval.accuracy['perAnswerType']:
        print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
    print("\n")

    # plot accuracy for various question types
    plt.bar(list(range(len(vqaEval.accuracy['perQuestionType']))), list(vqaEval.accuracy['perQuestionType'].values()),
            align='center')
    plt.xticks(list(range(len(vqaEval.accuracy['perQuestionType']))), list(vqaEval.accuracy['perQuestionType'].keys()),
               rotation='0', fontsize=10)
    plt.title('Per Question Type Accuracy', fontsize=10)
    plt.xlabel('Question Types', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.show()

    # sav evaluation results to ./Results folder

    json.dump(vqaEval.accuracy, open(os.path.join(output_dir, 'accuracy.json'), 'w+'))
    json.dump(vqaEval.evalQA, open(os.path.join(output_dir, 'eval_qa.json'), 'w+'))
    json.dump(vqaEval.evalQuesType, open(os.path.join(output_dir, 'ques_type.json'), 'w+'))
    json.dump(vqaEval.evalAnsType, open(os.path.join(output_dir, 'ans_type.json'), 'w+'))


def convert_to_vqa():
    # Convert our predictions into results for VQA
    path = paths.resources_path('predictions', 'beam_size_1', 'maxlen_20')
    predictions = os.listdir(path)

    print(predictions)

    for p in predictions:
        model_pred = os.path.join(path, p)
        with open(model_pred, 'r') as fp:
            p_data = json.load(fp)

        vqa_res = []

        print(model_pred)

        for q_id, ans in p_data.items():
            vqa_res.append({
                'question_id': int(q_id),
                'answer': ' '.join(ans)
            })

        with open(os.path.join(path, 'vqa_ready_{}'.format(p)), 'w+') as fp:
            json.dump(vqa_res, fp)

if __name__ == '__main__':
    convert_to_vqa()
