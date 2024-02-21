"""
This module is used for data ingestion and preprocessing. It reads raw data from a
specified path, preprocesses it, and saves the processed data to a specified path.
"""

from os.path import dirname, normpath

import pandas as pd

from src.constants import CONFIGS, SCHEMA
from src.exception import CustomException
from src.logger import logger
from src.utils.basic_utils import create_directories, read_yaml


class DataProcessor:
    """
    This class is responsible for data ingestion and preprocessing. It reads
    configuration and schema from yaml files, reads raw data from a specified path,
    preprocesses it, and saves the processed data to a specified path.
    """

    def __init__(self):
        # Read config files
        self.configs = read_yaml(CONFIGS).data_processor
        self.schema = read_yaml(SCHEMA)

        # Random seed for operations
        self.random_seed = self.configs.random_seed

        # Class names & column names
        self.class_names = self.schema.class_names
        self.column_names = self.schema.column_names

        # Input paths
        self.raw_filepath = normpath(self.configs.raw_data_path)

        # Output paths
        self.processed_filepath = normpath(self.configs.processed_data_path)

    @staticmethod
    def balance_dataframe(
        df: pd.DataFrame, class_names: list, label_column: str, random_seed: int
    ) -> pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            class_names (list): _description_
            label_column (str): _description_

        Returns:
            pd.DataFrame: _description_
        """
        class_counts = []
        for cls in class_names:
            n_rows = df[label_column].value_counts()[cls]
            class_counts.append((cls, n_rows))

        min_class = min(class_counts, key=lambda x: x[1])
        balanced_n = min_class[1]

        class_dfs_list = []
        for cls in class_names:
            part_df = df[df[label_column] == cls].sample(
                n=balanced_n, random_state=random_seed
            )
            class_dfs_list.append(part_df)

        balanced_df = (
            pd.concat(class_dfs_list, ignore_index=True)
            .sample(frac=1, random_state=random_seed)
            .reset_index(drop=True)
        )

        return balanced_df

    def preprocess_data(self) -> pd.DataFrame:
        """
        Reads the raw CSV data, renames the columns according to the schema, and
        returns the preprocessed data.

        Returns:
            pd.DataFrame: The preprocessed data.
        """
        # Read the raw TSV data
        df = pd.read_csv(
            self.raw_filepath,
            sep="\t",
            header=None,
            names=self.column_names,
            encoding="utf-8",
        )

        # Creating a balanced dataframe
        sms_df = self.balance_dataframe(
            df, self.class_names, self.column_names[0], self.random_seed
        )

        return sms_df

    def save_processed_data(self) -> None:
        """
        Creates a directory if it does not exist, performs data preprocessing, and
        saves the preprocessed data as a CSV file. If an exception occurs during this
        process, it is logged and re-raised.

        Raises:
            CustomException: If an error occurs during the data preprocessing or
            saving process.
        """
        try:
            # Create directory if not exits
            create_directories([dirname(self.processed_filepath)])

            # Perform data preprocessing
            logger.info("Ingest and Preprocess data")
            customers_df = self.preprocess_data()

            # Save the dataframe as CSV file
            customers_df.to_csv(
                self.processed_filepath, index=False, header=True, encoding="utf-8"
            )
            logger.info("Preprocessed data saved at: %s", self.processed_filepath)
        except Exception as e:
            logger.error(CustomException(e))
            raise CustomException(e) from e
