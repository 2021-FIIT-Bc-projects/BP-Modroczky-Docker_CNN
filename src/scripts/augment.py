import albumentations as a
import cv2
from pathlib import Path
import argparse
import json


def augment(aug_path, class_path, img_size):
    transform = a.Compose([
        a.Resize(
            height=img_size,
            width=img_size
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
        a.Resize(height=img_size, width=img_size)
    ])

    i = 0

    for image_path in class_path.iterdir():
        print("Processing {}".format(image_path))

        image = cv2.imread(image_path.as_posix())

        if image is None:
            exit("Image at {} cannot be read".format(image_path))

        # Copy original resized image
        cv2.imwrite(
            aug_path.joinpath("{}_{}.png".format(i, 0)).as_posix(),
            resize(image=image)['image']
        )

        range_end = count

        for j in range(1, range_end):
            # Write augmented image
            cv2.imwrite(
                aug_path.joinpath("{}_{}.png".format(i, j)).as_posix(),
                transform(image=image)['image']
            )
        i += 1


if __name__ == "__main__":
    img_size = 0
    classes = dict()
    train = dict()
    test = dict()
    data_path = ''
    augmented_data_path = ''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'jsonPath',
        type=str,
        help="Input json path that contains img_size integer, "
            "classes list, data_path, augmented_data_path"
    )
    args = parser.parse_args()
    print("Using json {}".format(Path(args.jsonPath)))

    try:
        with open(args.jsonPath) as json_file:
            data = json.load(json_file)
        classes = data.get('classes', dict())
        data_path = data.get('data_path', '')
        augmented_data_path = data.get('augmented_data_path', '')
        img_size = data.get('img_size', 0)
    except ValueError as err:
        exit("Cannot parse json from {}".format(args.jsonPath))

    if not classes or not data_path or not augmented_data_path or img_size == 0:
        exit("Json has to contain img_size integer, classes list, data_path, augmented_data_path")

    train = classes.get('train', dict())
    test = classes.get('test', dict())

    train_data_path = Path(data_path, 'train')
    train_augmented_data_path = Path(augmented_data_path, 'train')

    test_data_path = Path(data_path, 'test')
    test_augmented_data_path = Path(augmented_data_path, 'test')

    for class_name, count in train.items():
        class_path = Path(train_data_path, class_name)
        aug_path = Path(train_augmented_data_path, class_name)
        aug_path.mkdir(parents=True, exist_ok=True)

        if not class_path.exists():
            exit("Path {} does not exist".format(class_path))

        augment(aug_path, class_path, img_size)

    for class_name, count in test.items():
        class_path = Path(test_data_path, class_name)
        aug_path = Path(test_augmented_data_path, class_name)
        aug_path.mkdir(parents=True, exist_ok=True)

        if not class_path.exists():
            exit("Path {} does not exist".format(class_path))
        
        augment(aug_path, class_path, img_size)
