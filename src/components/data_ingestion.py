"""
summary
"""

from os.path import dirname, exists, normpath

import httpx

from src.constants import CONFIGS
from src.exception import CustomException
from src.logger import logger
from src.utils.basic_utils import create_directories, read_yaml, unzip_file


class DataIngestion:
    """
    summary
    """

    def __init__(self):
        """
        Initializes the DataIngestion class. Reads the configuration files.
        """
        # Read the configuration files
        self.configs = read_yaml(CONFIGS).data_ingestion

        # Source data configs
        self.data_url = self.configs.data_url
        self.user_agent = self.configs.user_agent
        self.timeout = self.configs.timeout
        self.download_status = self.configs.download_status

        # Output directories & filepath
        self.external_filepath = normpath(self.configs.external_path)
        self.raw_dir = normpath(self.configs.raw_dir)

    def download_data(self):
        """_summary_

        Raises:
            CustomException: _description_
        """
        try:
            if not exists(self.external_filepath) or self.download_status:
                # Create directory if not exist
                create_directories([dirname(self.external_filepath)])

                # Construct header
                headers = {"User-Agent": self.user_agent, "accept-language": "en-US"}

                # Download and save the data
                logger.info("File download started")
                with httpx.stream(
                    "GET", self.data_url, headers=headers, timeout=self.timeout
                ) as response:
                    with open(self.external_filepath, "wb") as file:
                        for chunk in response.iter_bytes():
                            file.write(chunk)
                logger.info("File downloaded successfully")
            else:
                logger.info(
                    "The %s already exists. Skipping download", self.external_filepath
                )
        except Exception as e:
            logger.error(CustomException(e))
            raise CustomException(e) from e

    def save_dataset(self):
        """_summary_

        Raises:
            CustomException: _description_
        """
        try:
            # Create directory if not exist
            create_directories([self.raw_dir])

            # Unzip the file
            logger.info("Unzipping the downloaded file")
            unzipped_files = unzip_file(
                zipfile_path=self.external_filepath, unzip_dir=self.raw_dir
            )
            logger.info("File unzipped. files are: %s", unzipped_files)
        except Exception as e:
            logger.error(CustomException(e))
            raise CustomException(e) from e
