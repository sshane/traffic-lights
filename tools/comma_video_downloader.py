import requests
import os
import time
from threading import Thread

try:
    from utils.JWT import JWT
    from utils.basedir import BASEDIR
except ImportError:
    BASEDIR = "C:/your_main_folder"  # any empty folder on your system
    JWT = "your_JWT_key"  # get your JWT from https://jwt.comma.ai/
    if BASEDIR == "C:/your_main_folder" or JWT == "your_JWT_key":
        raise Exception('Please fill in the BASEDIR and JWT variables at the top of this file.')


os.chdir(BASEDIR)


class CommaVideoDownloader:
    def __init__(self):
        """
            This tool allows multiple simultaneous downloads, simply enter your dongle ID and hit enter.
            To get your dongle ID, go to https://my.comma.ai/useradmin
        """

        self.new_data_folder = 'new_data'
        self.downloaded_dir = '{}/downloaded'.format(self.new_data_folder)
        self.video_extension = '.hevc'
        self.route_threads = []
        self.max_concurrent_downloads = 5
        self.setup_dirs()
        self.base_api_url = 'https://api.commadotai.com/v1/{}'

        self.start()

    def start(self):
        print('Paste dongle ID (ex. 3d5e08e90edb1c82)')
        dongle_id = input('>> ')
        dm = Thread(target=self.download_manager, args=(dongle_id,))
        dm.start()
        dm.join()

    def download_manager(self, dongle_id):
        response = requests.get(self.base_api_url.format('devices/{}/segments?from=0'.format(dongle_id)), headers={'Authorization': 'JWT {}'.format(JWT)})
        if response.status_code != 200:
            raise Exception('Error! Unknown status code: {}'.format(response.status_code))

        segments = response.json()
        routes = list(set([i['canonical_route_name'] for i in segments]))

        while len(routes) > 0:
            if len(self.route_threads) < self.max_concurrent_downloads:
                print('Starting download of route: {}'.format(routes[0]), flush=True)
                Thread(target=self.start_downloader, args=(routes[0],)).start()
                del routes[0]
                time.sleep(2)
            else:
                time.sleep(5)
            print('Currently downloading {} routes ({} left)...'.format(len(self.route_threads), len(routes)))

        while len(self.route_threads) > 0:
            print('Waiting for last few routes to finish downloading...')
            time.sleep(5)
        print('Finished downloading all available routes for this dongle!')


    def start_downloader(self, route_name):
        if route_name not in self.route_threads:
            self.route_threads.append(route_name)
        else:
            print('Thread already downloading route!')
            return

        sleep_time = 10
        successful_get = False
        for i in range(5):
            response = requests.get(self.base_api_url.format('route/{}/files').format(route_name), headers={'Authorization': 'JWT {}'.format(JWT)})
            if response.status_code == 200:
                successful_get = True
                break
            elif response.status_code == 429:
                # print('Too many requests, backing off and trying again (try: {})!'.format(i + 1))
                time.sleep(sleep_time)
                sleep_time **= 1.15
            else:
                print('{}: Unknown error!'.format(route_name))
                break

        if not successful_get:
            self.route_threads.remove(route_name)
            raise Exception('{}: Returned status code: {}'.format(route_name, response.status_code))


        response = response.json()
        video_urls = response['cameras']
        route_folder = self.get_name_from_url(video_urls[0])[1]
        self.make_dirs('{}/{}'.format(self.downloaded_dir, route_folder))

        for idx, video_url in enumerate(video_urls):
            # print('Downloading video {} of {}...'.format(idx + 1, len(video_urls)), flush=True)
            video_name = self.get_name_from_url(video_url)[0]

            video_save_path = '{}/{}/{}'.format(self.downloaded_dir, route_folder, video_name)
            if os.path.exists(video_save_path):
                # print('Video already downloaded: {}, skipping...'.format(video_name))
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

        self.route_threads.remove(route_name)

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
