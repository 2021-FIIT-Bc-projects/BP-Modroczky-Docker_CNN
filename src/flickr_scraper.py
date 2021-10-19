import flickrapi
import os
import dotenv
import json
import requests
from pathlib import Path

dotenv.load_dotenv('../.env')

api_key = os.getenv('FLICKR_API_KEY')
api_secret = os.getenv('FLICKR_API_SECRET')

flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')

per_page = 100
url = 'url_c'
extras = ['owner_name', 'date_taken', 'license', url]
queries = ['amanita muscaria', 'boletus']
dl_path = '../data/dataset_flickr/train'

for query in queries:
    first_page = flickr.photos.search(
        text=query,
        content_type=1,
        media='photos',
        license=4,
        per_page=per_page,
        page=1,
        extras=extras
    )

    all_photos = first_page['photos']['photo']
    num_of_pages = first_page['photos']['pages']

    for page in range(2, num_of_pages + 1):
        new_page = flickr.photos.search(
            text=query,
            content_type=1,
            media='photos',
            license=4,
            per_page=per_page,
            page=page,
            extras=extras
        )
        all_photos.extend(new_page['photos']['photo'])

    json_path = '../data/flickr_images_' + query.split(' ')[0] + '.json'
    with open(json_path, 'w') as output_file:
        json.dump({'photos': all_photos}, output_file)

    num = 0
    for photo in all_photos:
        if url not in photo:
            continue
        image = requests.get(photo[url], allow_redirects=False)
        filename = query.split(' ')[0] + '_' + str(num) + '.jpg'
        path = os.path.join(dl_path, query.split(' ')[0])
        Path(path).mkdir(parents=True, exist_ok=True)
        image_path = os.path.join(path, filename)
        open(image_path, 'wb').write(image.content)
        num += 1
