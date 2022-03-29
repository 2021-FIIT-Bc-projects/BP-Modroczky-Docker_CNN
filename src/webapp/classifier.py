from flask import Flask, render_template, request
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras.models import load_model
import numpy as np
import settings
from pathlib import Path

app = Flask(__name__)

@app.route('/')
def hello():
    return 'use /vgg16 or /inception-v3'


@app.route('/vgg16', methods=['GET', 'POST'])
def vgg16():
    if request.method == 'POST':
        model = load_model(Path(settings.models_path, settings.models['vgg16']['filename']))
        request.files['file'].save(Path(settings.media_root, 'tmp'))
        prediction = get_class(
            Path(settings.media_root, 'tmp').as_posix(), 
            model, 
            settings.models['vgg16']['class_names'],
            settings.models['vgg16']['img_size']
        )
        return render_template(
            'classifier.html',
            name="Classifier",
            data={"result": prediction, "model": "VGG16"}
        )
    
    return render_template('classifier.html', name="Classifier", data=None)


@app.route('/inception-v3', methods=['GET', 'POST'])
def inception_v3():
    if request.method == 'POST':
        model = load_model(Path(settings.models_path, settings.models['inception_v3']['filename']))
        request.files['file'].save(Path(settings.media_root, 'tmp'))
        prediction = get_class(
            Path(settings.media_root, 'tmp').as_posix(),
            model,
            settings.models['inception_v3']['class_names'],
            settings.models['inception_v3']['img_size']
        )
        return render_template(
            'classifier.html',
            name="Classifier",
            data={"result": prediction, "model": "Inception-v3"}
        )
    
    return render_template('classifier.html', name="Classifier", data=None)


def classify_image(model, image_path, img_size):
	image_data = load_img(
		image_path,
		target_size=(img_size, img_size),
		color_mode='rgb'
	)
	image = img_to_array(image_data).reshape((1, img_size, img_size, 3))
	return model.predict(image).flatten()


def get_class(image_path, model, class_names, img_size):
	each_class_probability = classify_image(model, image_path, img_size)

	final_label = np.argmax(each_class_probability, axis=0)
	final_probability = np.max(each_class_probability) * 100

	response = {'all': {class_names[i]: float(each_class_probability[i] * 100) for i in range(len(class_names))}}
	response['final'] = {class_names[final_label]: final_probability}

	return response


def save_file(file, media_root):
	Path(media_root).mkdir(parents=True, exist_ok=True)
	with open(media_root + '/tmp', 'wb+') as destination:
		for chunk in file.chunks():
			destination.write(chunk)
