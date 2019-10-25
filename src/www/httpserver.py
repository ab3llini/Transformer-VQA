import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir))
sys.path.append(root_path)

from flask import Flask, render_template, request, url_for
import shutil
from models.vgg_gpt2 import eval
import random
import matplotlib.pyplot as plt
import string
import os

app = Flask(__name__)
model, device, ts_dataset = eval.init_model_data()

cache = {'image': None, 'softmaps': None}


def randomString(stringLength=10):
    random.seed()
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


# Serve main index

def index(sample_id):
    random.seed()
    if sample_id == '' or sample_id is None:
        sample_id = random.randint(0, 200000)
    image = eval.get_sample_image(dataset=ts_dataset, index=sample_id)

    if cache['image'] is not None:
        os.remove('static/{}'.format(cache['image']))

    cache['image'] = randomString(10) + '.png'
    image.save("static/{}".format(cache['image']), "PNG")
    return render_template('index.html', sample_id=sample_id, image_href=url_for('static', filename=cache['image']))


@app.route('/')
def index_without_sample():
    return index(sample_id=None)


@app.route('/<sample_id>')
def index_with_sample(sample_id):
    return index(sample_id=int(sample_id))


@app.route('/execute')
def execute():
    question = request.args.get('question')
    sample_id = int(request.args.get('sample_id'))
    image, fig, words, alphas, sequence = eval.interactive_evaluation(question, model, device, ts_dataset, sample_id)
    if cache['softmaps'] is not None:
        os.remove('static/{}'.format(cache['softmaps']))
    cache['softmaps'] = randomString(11) + '.png'
    fig.savefig("static/{}".format(cache['softmaps']), dpi=150)
    return render_template('execute.html',
                           sample_id=sample_id,
                           image_href=url_for('static', filename=cache['image']),
                           output_href=url_for('static', filename=cache['softmaps']),
                           output=str(sequence)
                           )


if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0', debug=False, port=6006)
