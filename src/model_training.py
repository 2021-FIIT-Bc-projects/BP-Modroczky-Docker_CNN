import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

batch_size = 32
img_height = 180
img_width = 180
epochs = 100
seed = 27
checkpoint_path = 'models/cnn_amanita_boletus_mix'
training_data_path = 'data/dataset_amanita_boletus_mix/train'


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
    training_data = tf.keras.preprocessing.image_dataset_from_directory(
        directory=training_data_path,
        validation_split=0.2,
        subset='training',
        labels='inferred',
        label_mode='binary',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        seed=seed
    )
    validation_data = tf.keras.preprocessing.image_dataset_from_directory(
        directory=training_data_path,
        validation_split=0.2,
        subset='validation',
        labels='inferred',
        label_mode='binary',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        seed=seed
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

    data_augmentation = tf.keras.models.Sequential(
        [
            tf.keras.layers.RandomFlip(
                "horizontal",
                input_shape=(img_height, img_width, 3)
            ),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
        ]
    )

    model = tf.keras.models.Sequential([
        data_augmentation,
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_of_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.summary()

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        monitor='loss',
        mode='min',
        save_best_only=True
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8),
        model_checkpoint_callback
    ]

    history = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=callbacks
    )

    metrics_df = pd.DataFrame(history.history)

    metrics_df[["accuracy", "val_accuracy"]].plot()
    plt.title('Training and Validation Accuracy')

    metrics_df[["loss", "val_loss"]].plot()
    plt.title('Training and Validation Loss')

    plt.show()

    predict('data/dataset_amanita_boletus_mix/test/ama1.jpg')
    predict('data/dataset_amanita_boletus_mix/test/ama2.jpg')
    predict('data/dataset_amanita_boletus_mix/test/ama3.jpg')
    predict('data/dataset_amanita_boletus_mix/test/ama4.jpg')
    predict('data/dataset_amanita_boletus_mix/test/bol1.jpg')
    predict('data/dataset_amanita_boletus_mix/test/bol2.jpg')
    predict('data/dataset_amanita_boletus_mix/test/bol3.jpg')
    predict('data/dataset_amanita_boletus_mix/test/bol4.jpg')
