import requests
from utils.JWT import JWT
from utils.basedir import BASEDIR
import os
import time
from threading import Thread


# BASEDIR = "C:/your_main_folder"  # uncomment and fill these variables if you haven't cloned the entire traffic-lights repo. comment the related imports above
# JWT = "your_JWT_key"  # then comment the above imports for BASEDIR and JWT

os.chdir(BASEDIR)


class CommaVideoDownloader:
    def __init__(self):
        """
            This tool allows multiple simultaneous downloads, simply keep pasting route names and hit enter.
            Copying and pasting directly from your browser is supported, so if Chrome replaces | with %7C, it will still work.
            For example: 'e010b634f3d65cdb%7C2020-02-26--07-03-49'
            To get a list of drives, go to https://my.comma.ai/useradmin and click on your dongle id.
                The ids under `route_name` is what you want to paste here.
        """

        self.new_data_folder = 'new_data'
        self.downloaded_dir = '{}/downloaded'.format(self.new_data_folder)
        self.video_extension = '.hevc'
        self.setup_dirs()
        self.api_url = 'https://api.commadotai.com/v1/route/{}/files'
        self.download_loop()

    def download_loop(self):
        while True:
            print('Paste route name (ex. 23c2ed1a31ce0bda|2020-02-28--05-41-41)')
            route_name = input('>> ')
            Thread(target=self.start_downloader, args=(route_name,)).start()
            time.sleep(2)

    def start_downloader(self, route_name):
        response = requests.get(self.api_url.format(route_name), headers={'Authorization': 'JWT {}'.format(JWT)})
        if response.status_code != 200:
            raise Exception('Returned status code: {}'.format(response.status_code))

        response = response.json()
        video_urls = response['cameras']
        route_folder = self.get_name_from_url(video_urls[0])[1]
        self.make_dirs('{}/{}'.format(self.downloaded_dir, route_folder))

        print('Starting download...', flush=True)

        for idx, video_url in enumerate(video_urls):
            print('Downloading video {} of {}...'.format(idx + 1, len(video_urls)), flush=True)
            video_name = self.get_name_from_url(video_url)[0]

            video_save_path = '{}/{}/{}'.format(self.downloaded_dir, route_folder, video_name)
            if os.path.exists(video_save_path):
                print('Error, video already downloaded: {}, skipping...'.format(video_name))
                continue

            video = requests.get(video_url)  # don't download until we check if video already exists

            with open(video_save_path, 'wb') as f:
                f.write(video.content)
            if idx + 1 == len(video_urls):
                print('Successfully downloaded {} videos!'.format(len(video_urls)))
                break
            else:
                time.sleep(1)
        else:
            print('No new videos on this route! Please try again or wait until they have uploaded from your EON/C2.')

    def get_name_from_url(self, video_url):
        video_name = video_url.split('_')[-1]
        video_name = video_name[:video_name.index(self.video_extension) + len(self.video_extension)]
        return video_name, '--'.join(video_name.split('--')[:2])

    def make_dirs(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def setup_dirs(self):
        if not os.path.exists(self.downloaded_dir):
            os.makedirs(self.downloaded_dir)  # makes both folders recursively


video_downloader = CommaVideoDownloader()
