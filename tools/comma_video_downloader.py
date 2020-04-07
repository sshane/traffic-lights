import requests
import os
import time
from threading import Thread
import json

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
        self.download_db_file = '{}/downloaded/downloaded.json'.format(self.new_data_folder)
        self.video_extension = '.hevc'
        self.route_threads = []
        self.max_concurrent_downloads = 3
        self.max_retries = 6
        self.setup_dirs()
        self.base_api_url = 'https://api.commadotai.com/v1/{}'

        self.setup_db()
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
                print('\nStarting download of route: {}'.format(routes[0]), flush=True)
                Thread(target=self.start_downloader, args=(routes[0], dongle_id)).start()
                del routes[0]
                time.sleep(2)
            else:
                time.sleep(5)
            print('\nCurrently downloading {} routes ({} left)...'.format(len(self.route_threads), len(routes)))

        while len(self.route_threads) > 0:
            print('\nFinishing download of {} routes...'.format(len(self.route_threads)))
            time.sleep(5)
        print('Finished downloading all available routes for this dongle!')


    def start_downloader(self, route_name, dongle_id):
        try:
            if route_name not in self.route_threads:
                self.route_threads.append(route_name)
            else:
                print('Thread already downloading route!')
                return

            sleep_time = 10
            successful_get = False
            for i in range(self.max_retries):
                response = requests.get(self.base_api_url.format('route/{}/files').format(route_name), headers={'Authorization': 'JWT {}'.format(JWT)})
                if response.status_code == 200:
                    successful_get = True
                    break
                elif response.status_code == 429:
                    print('Too many requests, backing off and trying again (try: {} of {})!'.format(i + 1, self.max_retries))
                    time.sleep(sleep_time)
                    sleep_time = min(sleep_time ** 1.2, 120)
                else:
                    print('{}: Unknown error!'.format(route_name))
                    break

            if not successful_get:
                self.route_threads.remove(route_name)
                print('{}: Returned status code: {}'.format(route_name, response.status_code))
                return


            response = response.json()
            video_urls = response['cameras']
            if len(video_urls) == 0:
                print('Skipping empty route!')
                self.route_threads.remove(route_name)
                return
            route_folder = dongle_id + '_' + self.get_name_from_url(video_urls[0])[1]
            self.make_dirs('{}/{}'.format(self.downloaded_dir, route_folder))

            for idx, video_url in enumerate(video_urls):
                video_name = dongle_id + '_' + self.get_name_from_url(video_url)[0]
                video_save_path = '{}/{}/{}'.format(self.downloaded_dir, route_folder, video_name)

                if os.path.exists(video_save_path) or self.has_been_downloaded(dongle_id, video_name):
                    # print('Video already downloaded: {}, skipping...'.format(video_name))
                    self.update_db(dongle_id, video_name)  # add downloaded videos to db if not in db
                    continue
                print('Downloading video {} of {}...'.format(idx + 1, len(video_urls)), flush=True)

                video = requests.get(video_url)

                with open(video_save_path, 'wb') as f:
                    f.write(video.content)
                self.update_db(dongle_id, video_name)

                if idx + 1 == len(video_urls):
                    print('Successfully downloaded {} videos!'.format(len(video_urls)))
                    break
                else:
                    time.sleep(1)
            else:
                print('No new videos on this route! Please try again or wait until they have uploaded from your EON/C2.')
        except Exception as e:
            print('Error in video downloader!')
            print('Exception: {}'.format(e))

        if route_name in self.route_threads:
            self.route_threads.remove(route_name)

    def setup_db(self):
        if not os.path.exists(self.download_db_file):
            with open(self.download_db_file, 'w') as f:
                json.dump({}, f)

    def has_been_downloaded(self, dongle_id, video_name):
        with open(self.download_db_file, 'r') as f:
            downloads = json.load(f)
        return dongle_id in downloads and video_name in downloads[dongle_id]

    def update_db(self, dongle_id, video_name):
        with open(self.download_db_file, 'r') as f:
            downloads = json.load(f)

        if dongle_id not in downloads:
            downloads[dongle_id] = []
        if video_name not in downloads[dongle_id]:
            downloads[dongle_id].append(video_name)

        with open(self.download_db_file, 'w') as f:
            json.dump(downloads, f, indent=True)

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
