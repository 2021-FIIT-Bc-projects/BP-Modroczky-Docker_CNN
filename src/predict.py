import tensorflow as tf
import numpy as np
import os
import src.settings as settings


def predict():
    image_list = list()
    for directory in settings.class_names:
        class_path = os.path.join(settings.test_data_path, directory)
        for image in os.listdir(class_path):
            image_path = os.path.join(class_path, image)
            image_data = tf.keras.utils.load_img(
                image_path,
                target_size=(settings.img_height, settings.img_width),
                color_mode='rgb'
            )
            image_array = tf.keras.utils.img_to_array(image_data)
            image_list.append(image_array)
            print(image_array.shape)

    test_data_batch = np.array(image_list)

    print(test_data_batch.shape)
    predictions = model.predict(test_data_batch)
    print(predictions)

    for prediction in predictions:
        class_prediction = (prediction[0] > 0.5).astype("int32")
        class_name = settings.class_names[class_prediction]
        probability = 100 - (prediction[0] * 100) if class_name == 'amanita' else prediction[0] * 100
        print('This fungi image belongs to {} with probability of {:.2f}%'.format(class_name, probability))


model = tf.keras.models.load_model(settings.model_path)
predict()
