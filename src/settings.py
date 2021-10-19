batch_size = 64
img_height = 180
img_width = 180
epochs = 50
seed = 27

checkpoint_path = '../models/checkpoints/cnn_amanita_boletus.h5'
training_data_path = '../data/dataset_flickr_pixabay_combined/train'
test_data_path = '../data/dataset_flickr_pixabay_combined/test'
model_path = '../models/final_models/cnn_amanita_boletus.h5'
augmented_images_path = '../data/augmented_images'

class_names = ['amanita', 'boletus']
