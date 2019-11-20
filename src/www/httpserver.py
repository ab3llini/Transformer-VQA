import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir))
sys.path.append(root_path)

from flask import Flask, render_template, request, redirect
import models.vggpt2.interactive as vggpt2_it
import models.baseline.vqa.cyanogenoid.interactive as vqa_it
import www.utils.image as utils
import shutil
import hashlib

app = Flask(__name__)
image_folder = 'static/images/'

cols = 4
rows = 4


def get_md5_digest(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def init():
    prev_users = os.listdir(image_folder)
    for user in prev_users:
        shutil.rmtree(os.path.join(image_folder, user))


def get_session():
    return get_md5_digest(request.remote_addr)


def get_session_images():
    utils.delete_session_images(get_session())
    images = utils.k_rand_images(k=cols * rows)
    return utils.cache_session_images(images, get_session())


def get_answers_and_images(question, image_path):
    pil_image = utils.load_image(image_path)
    output = {}
    for model, it in zip(['VQABaseline', 'VGGPT-2'], [vqa_it, vggpt2_it]):
        answer, images = it.answer(question, pil_image)
        image_paths = []
        for i, image in enumerate(images):
            image_paths.extend(
                utils.cache_session_images({'{}.png'.format(get_md5_digest(model + str(i) + image_path)): image},
                                           get_session()))
            output[model] = {'answer': answer, 'images': image_paths}

    return output


@app.route('/')
def index():
    images = get_session_images()
    return render_template('index.html',
                           images=render_template('images.html', images=images, rows=rows, cols=cols))


@app.route('/upload/', methods=['post'])
def upload():
    session_md5 = get_session()
    endpoint = 'static'
    img_dir = os.path.join('images', session_md5)
    destination = os.path.join(img_dir, 'upload.png')
    files = request.files.getlist('image')
    if len(files) > 0:
        file = files[0]
        if not os.path.isdir(os.path.join(endpoint, img_dir)):
            os.mkdir(os.path.join(endpoint, img_dir))
        file.save(os.path.join(endpoint, destination))
        return redirect(os.path.join('/interact/', destination))
    else:
        return redirect('/')


@app.route('/interact/<dir>/<session>/<image>', methods=['GET', 'POST'])
def select_image(dir, session, image):
    curr_session = get_session()
    rel_path = os.path.join(dir, curr_session, image)
    path = os.path.join('static', rel_path)
    if not os.path.exists(path):
        # Allow other to share links
        # Load
        new = utils.load_relative_image(image)
        # Cache
        _ = utils.cache_session_images(new, curr_session)

    if session != curr_session:
        return redirect('/interact/' + rel_path)
    else:
        # Try to fetch a question
        question = request.form.get('question')
        print('Question =', question)
        answers = None
        if question is not None:
            print('Computing outputs..')
            answers = get_answers_and_images(question, path)
            print('Outputs:', answers)
        return interact(rel_path, answers)


@app.route('/interact/', methods=['GET', 'POST'])
def interact(image=None, answers=None):
    if image is not None:
        if answers is not None:
            return render_template('interact.html', target=image, answers=answers)
        else:
            return render_template('interact.html', target=image)
    else:
        return redirect('/')


if __name__ == '__main__':
    init()
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0', port=6006)
