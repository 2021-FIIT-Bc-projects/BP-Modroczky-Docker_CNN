import pandas as pd
import requests
import os
from pathlib import Path

df_path = '../data/mushroom_observer.tsv'
main_path = '../data/dataset/train'

classes = ['amanita muscaria', 'boletus', 'cantharellus', 'morchella']

df = pd.read_table(df_path)

print(df['license'].unique())

for class_name in classes:
    to_download = df[df['name'].astype(str).str.contains(class_name, case=False)].head(3000)

    print(len(to_download))
    names = set()
    num = 0

    path = os.path.join(main_path, class_name.split(' ')[0])
    Path(path).mkdir(parents=True, exist_ok=True)

    for _, row in to_download.iterrows():
        url = row['image']
        filename = "{}_{}_{}.{}".format(
            str(row['rightsHolder']).replace(' ', '_'),
            str(row['license']).split('/')[-3].replace('-', '_'),
            str(num),
            str(url).split('.')[-1]
        )
        names.add(row['rightsHolder'])
        try:
            image = requests.get(url, stream=True)
            image_path = os.path.join(path, filename)
            image_file = open(image_path, 'wb')
            image_file.write(image.content)
            image_file.close()
        except Exception as e:
            print(e)
        num += 1

    with open("../data/{}_authors.txt".format(class_name), 'w', encoding="utf-8") as f:
        for name in names:
            try:
                f.write("{}\n".format(str(name)))
            except Exception as e:
                print(e)
