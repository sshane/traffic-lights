import requests
from utils.JWT import JWT
from utils.basedir import BASEDIR
import os
import time

BASEDIR = os.path.join(BASEDIR, 'new_data')
downloaded_dir = '{}/downloaded'.format(BASEDIR)
os.chdir(BASEDIR)

video_extension = '.hevc'

route_name = input('Paste route name: ')
response = requests.get('https://api.commadotai.com/v1/route/{}/files'.format(route_name), headers={'Authorization': 'JWT {}'.format(JWT)})
if response.status_code != 200:
    raise Exception('Returned status code: {}'.format(response.status_code))

response = response.json()
video_urls = response['cameras']

if not os.path.exists(downloaded_dir):
    os.mkdir(downloaded_dir)

for idx, video_url in enumerate(video_urls):
    print('Downloading video {} of {}...'.format(idx, len(video_urls)))
    video_name = video_url.split('_')[-1]
    video_name = (video_name[:video_name.index(video_extension) + len(video_extension)])
    video = requests.get(video_url)
    with open('{}/{}'.format(downloaded_dir, video_name), 'wb') as f:
        f.write(video.content)
    time.sleep(2)

print('Successfully downloaded!')
