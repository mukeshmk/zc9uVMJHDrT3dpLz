import os
import shutil
import zipfile
import logging
import requests
import tempfile


logger = logging.getLogger(__name__)


def download_and_extract_zip() -> str:
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    temp_dir = tempfile.mkdtemp(prefix='zip_extract_')

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    
    zip_path = os.path.join(temp_dir, 'downloaded.zip')
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    os.remove(zip_path)

    return temp_dir
    

def remove_temp_dir(temp_dir: str) -> None:
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    extracted_path = download_and_extract_zip()
    print(extracted_path)
    remove_temp_dir(extracted_path)
    print("Temp directory removed")
