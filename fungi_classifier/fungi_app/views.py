from django.shortcuts import render
from rest_framework.views import APIView
from django.http import HttpResponse, JsonResponse
from tensorflow.keras.utils import img_to_array, load_img
from .apps import FungiAppConfig
import numpy as np
import os

img_height = 224
img_width = 224

class_names = ['amanita', 'boletus', 'cantharellus', 'morchella']
test_data_path = '../data/dataset_augmented/test'

def classify_image(class_names, model):
	image_path = os.path.join(test_data_path, 'amanita/ALP2010PIC55950385.jpg')
	image_data = load_img(
		image_path,
		target_size=(img_height, img_width),
		color_mode='rgb'
	)
	image = img_to_array(image_data).reshape((1, img_height, img_width, 3))
	return model.predict(image).flatten()

class call_model(APIView):
	def get(self, request):
		if request.method == 'GET':
			parameter = request.GET.get('sentence')
			each_class_probability = classify_image(class_names, FungiAppConfig.model)

			final_label = np.argmax(each_class_probability, axis=0)
			final_probability = np.max(each_class_probability) * 100

			response = {'all': {class_names[i]: float(each_class_probability[i]) for i in range(len(class_names))}}
			response['final'] = {class_names[final_label]: final_probability}

			return JsonResponse(response)