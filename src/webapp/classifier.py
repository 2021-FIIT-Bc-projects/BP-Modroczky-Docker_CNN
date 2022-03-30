from flask import Flask, render_template, request
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras.models import load_model
import numpy as np
import settings
from pathlib import Path

app = Flask(__name__)
model_vgg16 = load_model(Path(settings.models_path, settings.models['vgg16']['filename']))
model_inception_v3 = load_model(Path(settings.models_path, settings.models['inception_v3']['filename']))

@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/vgg16', methods=['GET', 'POST'])
def vgg16():
    if request.method == 'POST':
        return render_template(
            'classifier.html',
            name="Classifier",
            data={
                "result": run('vgg16', model_vgg16),
                "model": "VGG16"
            }
        )
    
    return render_template('classifier.html', name="Classifier", data=None)


@app.route('/inception-v3', methods=['GET', 'POST'])
def inception_v3():
    if request.method == 'POST':
        return render_template(
            'classifier.html',
            name="Classifier",
            data={
                "result": run('inception_v3', model_inception_v3),
                "model": "Inception-v3"
            }
        )
    
    return render_template('classifier.html', name="Classifier", data=None)


def run(model_name, model):
    request.files['file'].save(Path(settings.media_root, 'tmp'))
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
    response['final'] = {
        class_names[final_label]: final_probability
    }

    return response


def save_file(file, media_root):
	Path(media_root).mkdir(parents=True, exist_ok=True)
	with open(media_root + '/tmp', 'wb+') as destination:
		for chunk in file.chunks():
			destination.write(chunk)
