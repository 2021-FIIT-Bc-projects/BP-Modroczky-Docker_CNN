import os
from shutil import copyfile

path = "..\\data\\dataset_1394_classes\\train"
new_path = "..\\data\\dataset_418_classes\\train"

classes = set()

for folder in os.listdir(path):
    classes.add(folder.split('_')[1])

for one_class in classes:
    try:
        os.mkdir(os.path.join(new_path, one_class))
    except OSError as error:
        print(error)

for one_class in classes:
    for folder in os.listdir(path):
        if one_class in folder:
            for image in os.listdir(os.path.join(path, folder)):
                copyfile(os.path.join(path, folder, image), os.path.join(new_path, one_class, image))
                print("Copied image: {}".format(image))
