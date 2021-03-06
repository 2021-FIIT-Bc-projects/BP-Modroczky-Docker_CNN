from flask import Flask, render_template, request
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras.models import load_model
import numpy as np
import settings
from pathlib import Path

app = Flask(__name__)
model_vgg16 = load_model(Path(settings.models_path, settings.models['VGG16']['filename']))
model_inception_v3 = load_model(Path(settings.models_path, settings.models['Inception-v3']['filename']))

@app.route('/')
def hello():
    return render_template('index.html', name='Models')


@app.route('/vgg16', methods=['GET', 'POST'])
def vgg16():
    name = 'VGG16'
    if request.method == 'POST':
        return render_template(
            'classifier.html',
            name=name,
            data={
                "result": run(name, model_vgg16),
                "edibility": settings.edibility
            }
        )
    return render_template('classifier.html', name=name, data=None)


@app.route('/inception-v3', methods=['GET', 'POST'])
def inception_v3():
    name = 'Inception-v3'
    if request.method == 'POST':
        return render_template(
            'classifier.html',
            name=name,
            data={
                "result": run(name, model_inception_v3),
                "edibility": settings.edibility
            }
        )
    return render_template('classifier.html', name=name, data=None)


def run(model_name, model):
    media_path = Path(settings.media_root)
    media_path.mkdir(parents=True, exist_ok=True)
    request.files['file'].save(Path(media_path, 'tmp'))
    prediction = classify(
        Path(settings.media_root, 'tmp').as_posix(),
        model,
        settings.models[model_name]['class_names'],
        settings.models[model_name]['img_size']
    )
    return prediction


def classify(image_path, model, class_names, img_size):
    image_data = load_img(image_path, target_size=(img_size, img_size), color_mode='rgb')
    image = img_to_array(image_data).reshape((1, img_size, img_size, 3))
    each_class_probability = model.predict(image).flatten()

    final_label = np.argmax(each_class_probability, axis=0)
    final_probability = round(np.max(each_class_probability) * 100, 2)

    response = {
        'all': {
                class_names[i]: 
                    round(each_class_probability[i] * 100, 2)
                    for i in range(len(class_names))
            }
        }
    response['final_class'] = class_names[final_label]
    response['final_probability'] = final_probability

    return response
