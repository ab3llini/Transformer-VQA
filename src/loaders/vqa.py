import os
from vqaTools.vqa import VQA
import random
import matplotlib.pyplot as plt
import skimage.io as io
from tqdm import tqdm

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


# High level representation of a question
# Contains question, image and annotated answers.
class Question:
    def __init__(self, _id, question, image_path, answers):
        self._id = _id
        self.question = question
        self.image_path = image_path
        self.answers = answers

    def plot(self):
        img = self.load_image()
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def load_image(self):
        return io.imread(self.image_path)

    def __str__(self):
        return 'ID: {}\nQuestion: {}\nAnswers: {}\nImage {}'.format(self._id, self.question, self.answers, self.image_path)


class VQALoader:
    def __init__(self):
        self.vqa = VQA(annFile, quesFile)
        self.questions = []

    def get_questions(self):
        # Load all question ids
        ids = self.vqa.getQuesIds()
        # Get QA instances
        qa = self.vqa.loadQA(ids)

        print('Building question structure..')

        # Translate to our own structure
        for _qa in tqdm(qa):
            id = _qa['question_id']
            # Access actual question rather than it's ID
            question = self.vqa.qqa[id]['question']
            # Pre compute actual image path
            image_path = imgDir + 'COCO_' + dataSubType + '_' + str(_qa['image_id']).zfill(12) + '.jpg'
            # Extract only answers and get rid of additional annotations
            answers = [q['answer'] for q in _qa['answers']]

            self.questions.append(Question(id, question, image_path, answers))

        return self.questions

    # Picks qty questions randomly for general purpose usage.
    def random_samples(self, qty=1):
        if len(self.questions) == 0:
            self.get_questions()
        return random.sample(self.questions, qty)



if __name__ == '__main__':
    loader = VQALoader()

    sample = loader.random_samples()[0]

    sample.plot()
    print(sample)

