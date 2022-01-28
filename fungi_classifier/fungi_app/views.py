from django.shortcuts import render
from rest_framework.views import APIView
from django.http import HttpResponse, JsonResponse
from tensorflow.keras.utils import img_to_array, load_img
from .apps import FungiAppConfig
import numpy as np
import os
from fungi_classifier import settings

def classify_image(class_names, model):
	image_path = os.path.join(settings.test_data_path, 'amanita/ALP2010PIC55950385.jpg') # will use a variable image name
	image_data = load_img(
		image_path,
		target_size=(settings.img_height, settings.img_width),
		color_mode='rgb'
	)
	image = img_to_array(image_data).reshape((1, settings.img_height, settings.img_width, 3))
	return model.predict(image).flatten()

class call_model(APIView):
	def get(self, request):
		if request.method == 'GET':
			parameter = request.GET.get('sentence')
			each_class_probability = classify_image(settings.class_names, FungiAppConfig.model)

			final_label = np.argmax(each_class_probability, axis=0)
			final_probability = np.max(each_class_probability) * 100

			response = {'all': {settings.class_names[i]: float(each_class_probability[i]) for i in range(len(settings.class_names))}}
			response['final'] = {settings.class_names[final_label]: final_probability}

			return JsonResponse(response)