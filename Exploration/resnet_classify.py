from VQA.PythonHelperTools.vqaTools import vqa as vqa
from VQA import config as vqa_config
from os import listdir
import os
import numpy as np
from tqdm import tqdm
import random
from keras.applications import resnet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import numpy as np

# vqa_api = vqa.VQA(vqa_config.annFile, vqa_config.quesFile)

# q_ids = vqa_api.getQuesIds()
# qa = vqa_api.loadQA(q_ids)

# vqa_api.showQA(random.sample(qa, 10))

# Predict image with resnet
# imgId = random.choice(qa)['image_id']
imgId = 9
imgFileName = vqa_config.imgDir + 'COCO_' + vqa_config.dataSubType + '_' + str(imgId).zfill(12) + '.jpg'

loaded_image = load_img(imgFileName, target_size=(224, 224))
numpy_image = img_to_array(loaded_image)
input_image = np.expand_dims(numpy_image, axis=0)

print('PIL image size = ', loaded_image.size)
print('NumPy image size = ', numpy_image.shape)
print('Input image size = ', input_image.shape)
plt.imshow(np.uint8(input_image[0]))
plt.show()

processed_image_resnet50 = resnet50.preprocess_input(input_image.copy())


resnet_model = resnet50.ResNet50(weights='/opt/project/ImageNet/Models/resnet50.h5')
predictions_resnet50 = resnet_model.predict(processed_image_resnet50)
label_resnet50 = decode_predictions(predictions_resnet50)
print ('label_resnet50 = ', label_resnet50)