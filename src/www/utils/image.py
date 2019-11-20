import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

import random
from utilities import paths
from PIL import Image
import hashlib


def load_image(path):
    fp = open(path, 'rb')
    return Image.open(fp)


def k_rand_images(k, directory=paths.data_path('vqa', 'Images', 'val2014')):
    samples, full_paths = k_rand_image_paths(k, directory)

    return {sample: load_image(path) for sample, path in zip(samples, full_paths)}


def load_relative_image(name, directory=paths.data_path('vqa', 'Images', 'val2014')):
    return {name: load_image(os.path.join(directory, name))}


def k_rand_image_paths(k, directory):
    # Get all images in directory
    population = os.listdir(directory)

    # Sample K images
    samples = random.sample(population, k)

    return samples, list(map(lambda image: os.path.join(directory, image), samples))


def cache_session_images(images, session, endpoint='static', directory='images'):
    session = os.path.join(directory, session)
    destination = os.path.join(endpoint, session)
    if not os.path.exists(destination):
        os.makedirs(destination)
    image_names = []
    for img_path, img in images.items():
        session_image = '{}'.format(os.path.join(session, img_path))
        image_names.append(session_image)
        _, ext = os.path.splitext(session_image)
        ext = ext[1:]
        ext = 'JPEG' if ext.lower() == 'jpg' else ext.upper()
        img.save(os.path.join(endpoint, session_image), ext)

    return image_names


def delete_session_images(session, endpoint='static', directory='images'):
    session = os.path.join(directory, session)
    destination = os.path.join(endpoint, session)
    if os.path.exists(destination):
        image_names = os.listdir(destination)
        for session_image in image_names:
            os.remove(os.path.join(destination, session_image))
