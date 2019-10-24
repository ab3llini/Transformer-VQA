import matplotlib.pyplot as plt
from utilities.vqa.dataset import *
from utilities.vqa.vqa import VQA
from utilities.vqa.evaluation import VQAEval


# An example result json file has been provided in './Results' folder.

def vqa_evaluation(questions, annotations, results, output_dir, result_ques_ids=None, precision=2):
    # create vqa object and vqaRes object
    vqa = VQA(annotations, questions)
    vqaRes = vqa.loadRes(results, questions)

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

    # save evaluation results to ./Results folder
    json.dump(vqaEval.accuracy, open(os.path.join(output_dir, 'accuracy'), 'w+'))
    json.dump(vqaEval.evalQA, open(os.path.join(output_dir, 'eval_qa'), 'w+'))
    json.dump(vqaEval.evalQuesType, open(os.path.join(output_dir, 'ques_type'), 'w+'))
    json.dump(vqaEval.evalAnsType, open(os.path.join(output_dir, 'ans_type'), 'w+'))
