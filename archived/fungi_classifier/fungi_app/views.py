from django.shortcuts import render
from rest_framework.views import APIView
from django.http import HttpResponse, JsonResponse
from tensorflow.keras.utils import img_to_array, load_img
from .apps import FungiAppConfig
from .forms import UploadFileForm
import numpy as np
import os
from fungi_classifier import settings
import json
from pathlib import Path


def classify_image(model, image_path):
	image_path = os.path.join(image_path)
	image_data = load_img(
		image_path,
		target_size=(settings.IMG_SIZE, settings.IMG_SIZE),
		color_mode='rgb'
	)
	image = img_to_array(image_data).reshape((1, settings.IMG_SIZE, settings.IMG_SIZE, 3))
	return model.predict(image).flatten()


def get_class(tmp):
	image_path = tmp
	each_class_probability = classify_image(FungiAppConfig.model, image_path)

	final_label = np.argmax(each_class_probability, axis=0)
	final_probability = np.max(each_class_probability) * 100

	response = {'all': {settings.CLASS_NAMES[i]: float(each_class_probability[i] * 100) for i in range(len(settings.CLASS_NAMES))}}
	response['final'] = {settings.CLASS_NAMES[final_label]: final_probability}

	return response


def save_file(file):
	Path(settings.MEDIA_ROOT).mkdir(parents=True, exist_ok=True)
	with open(settings.MEDIA_ROOT + '/tmp', 'wb+') as destination:
		for chunk in file.chunks():
			destination.write(chunk)


def upload_file(request):
	if request.method == 'POST':
		form = UploadFileForm(request.POST, request.FILES)
		if form.is_valid():
			form.save()
			save_file(request.FILES['file'])
			return render(
				request, 
				'classifier_gui.html', 
				{
					'form': UploadFileForm(),
					'data': get_class(settings.MEDIA_ROOT + '/tmp'),
					'image': form.instance
				}
			)
	else:
		form = UploadFileForm()
	return render(request, 'classifier_gui.html', {'form': form})
