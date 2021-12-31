import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model

model_path = '../models/final_models/vgg16.h5'
test_data_path = '../data/dataset_augmented/test'

batch_size = 64
img_height = 224
img_width = 224

class_names = ['amanita', 'boletus', 'cantharellus', 'morchella']

model = load_model(model_path)


def predict_and_evaluate(images, labels, class_names):
    test_data_batch = np.array(images)
    print(test_data_batch.shape)

    predictions = model.predict(test_data_batch)
    predicted_labels = np.argmax(predictions, axis=1)

    hits = 0

    for predicted_label, prediction, label in zip(predicted_labels, predictions, labels):
        probability = np.max(prediction) * 100
        print(
            "{} with {:.2f}% probability (real class: {})".format(
                class_names[predicted_label],
                probability,
                class_names[label]
            )
        )

        hits = hits + 1 if label == predicted_label else hits

    accuracy = (hits / len(labels)) * 100
    return accuracy, predicted_labels


images = list()
labels = list()

for class_name in class_names:
    class_path = os.path.join(test_data_path, class_name)
    for image in os.listdir(class_path):
        image_path = os.path.join(class_path, image)
        image_data = load_img(
            image_path,
            target_size=(img_height, img_width),
            color_mode='rgb'
        )
        image_array = img_to_array(image_data)
        images.append(image_array)
        labels.append(class_names.index(class_name))

accuracy, predicted_labels = predict_and_evaluate(images, labels, class_names)
print("Accuracy is {:.2f}%".format(accuracy))

evaluation = model.evaluate(
    x=np.array(images),
    y=np.array(labels),
    batch_size=batch_size
)
print("Loss is {:.4f}".format(evaluation[0]))
print("Accuracy is {:.2f}%".format(100 * evaluation[1]))

print(classification_report(labels, predicted_labels, target_names=class_names))

matrix = confusion_matrix(labels, predicted_labels)

sns.heatmap(
    matrix,
    square=True,
    annot=True,
    cbar=False,
    cmap=plt.cm.Blues,
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel('True Classes')
plt.ylabel('Predicted Classes')
plt.title('Confusion Matrix')
plt.savefig('../plots/confusion_matrix.png')
plt.show()
