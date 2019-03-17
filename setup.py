import requests
from pathlib import Path
import time
import os

def fetch_and_cache(data_url, file, data_dir="data", force=False):
    """
    Download and cache a url and return the file object.

    data_url: the web address to download
    file: the file in which to save the results.
    data_dir: (default="data") the location to save the data
    force: if true the file is always re-downloaded

    return: The pathlib.Path object representing the file.

    EXAMPLE USAGE:
    data_url = 'https://s3-us-west-2.amazonaws.com/multimedia-berkeley/Flickr.tar.gz',
    data_directory = './data'
    data_filename = 'nus-wide.tar.gz',
    fetch_and_cache(data_url = data_url, data_dir = data_directory, file = data_filename, force = False)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok = True)
    file_path = data_dir / Path(file)
    if force and file_path.exists():
        file_path.unlink()
    if force or not file_path.exists():
        print('Downloading...', end=' ')
        resp = requests.get(data_url)
        with file_path.open('wb') as f:
            f.write(resp.content)
        print('Done!')
        last_modified_time = time.ctime(file_path.stat().st_mtime)
    else:
        last_modified_time = time.ctime(file_path.stat().st_mtime)
        print("Using cached version that was downloaded (UTC):", last_modified_time)
    return file_path


fetch_and_cache(data_url='https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz',
                file='writingPrompts.tar.gz',
                data_dir='./data',
                force=False)

os.system("tar -xvzf ./data/writingPrompts.tar.gz -C ./data")
os.system("rm ./data/writingPrompts.tar.gz")
