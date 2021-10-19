import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import src.settings as settings

if __name__ == '__main__':
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.3,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
        brightness_range=(0.6, 1.4),
        rotation_range=20
    )

    training_generator = data_generator.flow_from_directory(
        directory=settings.training_data_path,
        subset='training',
        classes=settings.class_names,
        class_mode='binary',
        batch_size=settings.batch_size,
        target_size=(settings.img_height, settings.img_width),
        seed=settings.seed,
        shuffle=True
    )

    validation_generator = data_generator.flow_from_directory(
        directory=settings.training_data_path,
        subset='validation',
        classes=settings.class_names,
        class_mode='binary',
        batch_size=settings.batch_size,
        target_size=(settings.img_height, settings.img_width),
        seed=settings.seed,
        shuffle=True
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            16,
            3,
            padding='same',
            activation='relu',
            input_shape=(settings.img_height, settings.img_width, 3)
        ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(
            32,
            3,
            padding='same',
            activation='relu'
        ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(
            64,
            3,
            padding='same',
            activation='relu'
        ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    model.summary()

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=settings.checkpoint_path,
        save_weights_only=False,
        monitor='loss',
        mode='min',
        save_best_only=True
    )

    callbacks = [
        model_checkpoint_callback
    ]

    history = model.fit(
        training_generator,
        validation_data=validation_generator,
        epochs=settings.epochs,
        callbacks=callbacks
    )

    metrics_df = pd.DataFrame(history.history)

    metrics_df[["accuracy", "val_accuracy"]].plot()
    plt.title('Training and Validation Accuracy')
    plt.show()

    metrics_df[["loss", "val_loss"]].plot()
    plt.title('Training and Validation Loss')
    plt.show()

    model.save('../models/final_models/cnn_amanita_boletus.h5')
