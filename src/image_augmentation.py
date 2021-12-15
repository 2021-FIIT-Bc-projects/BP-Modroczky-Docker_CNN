import albumentations as a
import cv2
import os
from pathlib import Path

classes = ['amanita', 'boletus', 'cantharellus', 'morchella']

img_height = 224
img_width = 224

training_data_path = '../data/dataset/train'
augmented_images_path = '../data/dataset_augmented/train'

transform = a.Compose([
    a.Resize(
        height=img_height,
        width=img_width
    ),
    a.HorizontalFlip(),
    a.RandomBrightnessContrast(
        p=1,
        contrast_limit=(-0.2, 0.2),
        brightness_limit=(-0.2, 0.2)
    ),
    a.Affine(
        p=1,
        mode=cv2.BORDER_REPLICATE
    )
])

resize = a.Compose([
    a.Resize(height=img_height, width=img_width)
])

for class_name in classes:
    class_path = os.path.join(training_data_path, class_name)
    i = 0
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        image = cv2.imread(image_path)
        print("processing {}".format(image_path))

        aug_path = os.path.join(augmented_images_path, class_name)
        Path(aug_path).mkdir(parents=True, exist_ok=True)
        # Copy original resized image
        cv2.imwrite(os.path.join(aug_path, "{}_{}.png".format(i, 0)), resize(image=image)['image'])

        range_end = 58
        if class_name == 'boletus':
            range_end = 48
        elif class_name == 'cantharellus':
            range_end = 49
        elif class_name == 'morchella':
            range_end = 46

        for j in range(1, range_end):
            # Write augmented image
            cv2.imwrite(os.path.join(aug_path, "{}_{}.png".format(i, j)), transform(image=image)['image'])
        i += 1
