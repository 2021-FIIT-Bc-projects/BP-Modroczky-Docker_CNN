from django.apps import AppConfig
from tensorflow.keras.models import load_model

class FungiAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fungi_app'
    model_path = '../models/final_models/vgg16.h5'
    model = load_model(model_path)
