import pandas as pd
import requests
from pathlib import Path
import argparse
import json

headers = {
    'User-Agent':
        'Mozilla/5.0 (Windows NT 6.1; WOW64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/56.0.2924.76 Safari/537.36'
}

limit = 0
tsv_path = ''
dl_path = ''
authors_path = ''
queries = list()

parser = argparse.ArgumentParser()
parser.add_argument(
    'jsonName',
    type=str,
    help="Input name of JSON that contains classes list, tsv_path, dl_path"
)

args = parser.parse_args()
print("Using JSON {}".format(args.jsonName))

try:
    with Path('..', '..', "metadata", args.jsonName).open() as json_file:
        data = json.load(json_file)
    queries = data.get('queries', dict())
    tsv_path = data.get('tsv_path', '')
    dl_path = data.get('dl_path', '')
    authors_path = data.get('authors_path', '')
    limit = data.get('limit', 0)
except ValueError:
    exit("Cannot parse data from {}".format(args.jsonName))
except FileNotFoundError:
    exit("JSON {} not found".format(args.jsonName))

if not queries or not tsv_path or not dl_path or not authors_path or limit == 0:
    exit("JSON has to contain classes list, tsv_path, dl_path")

df = pd.read_table(Path('..', '..', tsv_path))

for query in queries:
    to_download = df[df['name'].astype(str).str.contains(query, case=False)].head(limit)

    print("Downloading {} pictures of {} with limit {}".format(len(to_download), query, limit))
    names = set()
    num = 0

    path = Path('..', '..', dl_path, query.split(' ')[0])
    path.mkdir(parents=True, exist_ok=True)

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
            print("Downloading {}".format(url))
            image = requests.get(url, stream=True, headers=headers)
            print("Status code {}".format(image.status_code))
            image_path = Path(path, filename)
            with open(image_path, 'wb+') as f:
                f.write(image.content)
        except Exception as e:
            print(e)
        num += 1

    with open(Path('..', '..', authors_path, "{}_authors.txt".format(query)), 'w', encoding="utf-8") as f:
        for name in names:
            try:
                f.write("{}\n".format(str(name)))
            except Exception as e:
                print(e)
