import requests
import os
from pathlib import Path
import dotenv
import decimal

dotenv.load_dotenv('../.env')

api = 'https://pixabay.com/api/'
api_key = os.getenv('PIXABAY_API_KEY')

dl_path = '../data/dataset_pixabay/train'

queries = ['amanita', 'boletus']

lang = 'en'
image_type = 'photo'
category = 'nature'

per_page = 100

parameters = {
    'key': api_key,
    'image_type': image_type,
    'page': 1,
    'per_page': per_page
}

for query in queries:
    image_urls = list()
    parameters['q'] = query
    parameters['page'] = 1

    first_page = requests.get(api, params=parameters).json()

    for image in first_page['hits']:
        image_urls.append(image['webformatURL'])

    num_of_pages = decimal.Decimal(
        first_page['totalHits'] / per_page
    ).quantize(
        decimal.Decimal('1.'),
        rounding=decimal.ROUND_UP
    )

    for page in range(2, int(num_of_pages) + 1):
        parameters['page'] = page
        response = requests.get(api, params=parameters).json()
        for image in response['hits']:
            image_urls.append(image['webformatURL'])

    num = 0
    for image_url in image_urls:
        image = requests.get(image_url, allow_redirects=False)
        filename = query + '_' + str(num) + '.jpg'
        path = os.path.join(dl_path, query)
        Path(path).mkdir(parents=True, exist_ok=True)
        image_path = os.path.join(path, filename)
        open(image_path, 'wb').write(image.content)
        num += 1
