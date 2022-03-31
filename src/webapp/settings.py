models = {
    "vgg16": {
        "filename": "vgg16.h5",
        "img_size": 224,
        "class_names": [
            'amanita',
            'boletus',
            'cantharellus',
            'morchella'
        ]
    },
    "inception_v3": {
        "filename": "inception_v3.h5",
        "img_size": 299,
        "class_names": [
            'amanita',
            'boletus',
            'cantharellus',
            'craterellus',
            'macrolepiota',
            'morchella',
            'pleurotus',
            'psilocybe'
        ]
    }
}

edibility = {
    "amanita": False,
    "boletus": True,
    "cantharellus": True,
    "craterellus": True,
    "macrolepiota": True,
    "morchella": True,
    "pleurotus": True,
    "psilocybe": False
}

media_root = "static/media"

models_path = "../../models"

allowed_extension = { 'png', 'jpg', 'jpeg' }