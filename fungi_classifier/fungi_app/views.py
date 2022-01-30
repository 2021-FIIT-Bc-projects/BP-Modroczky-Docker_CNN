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


def classify_image(class_names, model, image_path):
	image_path = os.path.join(image_path)
	image_data = load_img(
		image_path,
		target_size=(settings.img_height, settings.img_width),
		color_mode='rgb'
	)
	image = img_to_array(image_data).reshape((1, settings.img_height, settings.img_width, 3))
	return model.predict(image).flatten()


def get_class(tmp):
	image_path = tmp
	each_class_probability = classify_image(settings.class_names, FungiAppConfig.model, image_path)

	final_label = np.argmax(each_class_probability, axis=0)
	final_probability = np.max(each_class_probability) * 100

	response = {'all': {settings.class_names[i]: float(each_class_probability[i]) for i in range(len(settings.class_names))}}
	response['final'] = {settings.class_names[final_label]: final_probability}

	return response


def save_file(file):
	with open(settings.uploaded_img_path + '/tmp', 'wb+') as destination:
		for chunk in file.chunks():
			destination.write(chunk)


def upload_file(request):
	if request.method == 'POST':
		form = UploadFileForm(request.POST, request.FILES)
		if form.is_valid():
			save_file(request.FILES['file'])
			return render(request, 'result.html', {'data': get_class(settings.uploaded_img_path + '/tmp')})
	else:
		form = UploadFileForm()
	return render(request, 'classifier_gui.html', {'form': form})
