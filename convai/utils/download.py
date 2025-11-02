import os
import shutil
import zipfile
import logging
import requests
import tempfile

from convai.utils.config import settings
from convai.utils.logger import setup_logs


logger = logging.getLogger(__name__)


def download_and_extract_zip() -> str:
    """
    Downloads a zip file from the provided URL into a temporary directory,
    extracts its contents, and returns the path to the extracted data.
    
    Args:
        url: The URL of the zip file to download
        
    Returns:
        str: Path to the temporary directory containing extracted files
        
    Raises:
        requests.RequestException: If download fails
        zipfile.BadZipFile: If the file is not a valid zip file
    """
    url = settings.MOVIELENS_DOWNLOAD_URL
    # Create a temporary directory that won't be auto-deleted
    temp_dir = tempfile.mkdtemp(prefix='zip_extract_')
    logger.debug(f"created temp directory: {temp_dir}")
    
    try:
        # Download the zip file
        logger.info(f"Downloading from: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Save to temporary zip file
        zip_path = os.path.join(temp_dir, 'downloaded.zip')
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logger.debug(f"Download complete. Extracting")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Remove the zip file after extraction to save space
        os.remove(zip_path)
        
        logger.info(f"Extraction complete. Files extracted to: {temp_dir}")
        return temp_dir
    
    except requests.RequestException as e:
        logger.error(f"Download failed: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e
    except zipfile.BadZipFile:
        logger.error("The downloaded file is not a valid zip file")
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e


def remove_temp_dir(temp_dir: str) -> None:
    shutil.rmtree(temp_dir, ignore_errors=True)
    logger.info(f"remove the temp folder at: {temp_dir}")


if __name__ == "__main__":
    setup_logs(cli_level="debug")
    
    extracted_path = download_and_extract_zip()
    logger.info(f"Data files are located at: {extracted_path}")
        
