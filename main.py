import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np

batch_size = 32
img_height = 180
img_width = 180

training_data_path = 'dataset_mix/train'
# test_data_path = 'dataset/test'


def predict(img_path):
    img = tf.keras.utils.load_img(
        img_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print("This image most likely belongs to {} with a {:.2f} percent confidence."
          .format(class_names[np.argmax(score)], 100 * np.max(score)))


if __name__ == '__main__':
    training_data = image_dataset_from_directory(
        directory=training_data_path,
        validation_split=0.2,
        subset='training',
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        seed=274
    )
    validation_data = image_dataset_from_directory(
        directory=training_data_path,
        validation_split=0.2,
        subset='validation',
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        seed=274
    )

    class_names = training_data.class_names
    num_of_classes = len(class_names)

    autotune = tf.data.AUTOTUNE

    training_data = training_data.cache().shuffle(1000).prefetch(buffer_size=autotune)
    validation_data = validation_data.cache().prefetch(buffer_size=autotune)

    for image_batch, labels_batch in training_data:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    data_augmentation = Sequential(
        [
            layers.RandomFlip(
                "horizontal",
                input_shape=(img_height, img_width, 3)
            ),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_of_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.summary()

    epochs = 10
    history = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=epochs
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    predict('dataset_mix/test/ama1.jpg')
    predict('dataset_mix/test/ama2.jpg')
    predict('dataset_mix/test/ama3.jpg')
    predict('dataset_mix/test/ama4.jpg')
    predict('dataset_mix/test/bol1.jpg')
    predict('dataset_mix/test/bol2.jpg')
    predict('dataset_mix/test/bol3.jpg')
    predict('dataset_mix/test/bol4.jpg')

    model.save('saved_model/my_model')
