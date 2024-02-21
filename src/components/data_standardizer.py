"""
This module provides a class for data transformation, which includes
reading configuration files, separating numerical and categorical features,
constructing a preprocessor for normalization, and transforming train and test data.

Classes:
    DataTransformation: A class for transforming data.
"""

import string
from os.path import dirname, normpath

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from scipy.sparse import hstack, save_npz
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from src.constants import CONFIGS, SCHEMA
from src.exception import CustomException
from src.logger import logger
from src.utils.basic_utils import create_directories, read_yaml, save_as_joblib

# Download stropwords from nltk library
nltk.download("stopwords")


class DataStandardizer:
    """
    This class is responsible for transforming raw data into a format suitable for
    machine learning models. It reads configuration and schema files, constructs a
    preprocessor for normalization of numerical features, and applies this preprocessor
    to transform the train and test data. The transformed data is then saved as numpy
    arrays for further use. The class also handles the creation of necessary
    directories and the saving of the preprocessor object for future use.
    """

    def __init__(self):
        # Read the configuration files
        self.configs = read_yaml(CONFIGS).data_standardizer
        self.schema = read_yaml(SCHEMA)

        # Class names & column names
        self.class_names = self.schema.class_names
        self.column_names = self.schema.column_names

        # Input file paths
        self.train_data_path = normpath(self.configs.train_data_path)
        self.test_data_path = normpath(self.configs.test_data_path)

        # Output file paths
        self.train_matrix_path = normpath(self.configs.train_matrix_path)
        self.test_matrix_path = normpath(self.configs.test_matrix_path)
        self.standardizer_path = normpath(self.configs.standardizer_path)

    @staticmethod
    def get_tokens(texts):
        """_summary_

        Args:
            texts (_type_): _description_

        Returns:
            _type_: _description_
        """
        en_stopwords = stopwords.words("english")
        msg_tokens = []

        def extract_tokens(msg):
            nopunc_words = "".join(
                [char for char in msg if char not in string.punctuation]
            ).split()
            tokens = [word for word in nopunc_words if word.lower() not in en_stopwords]
            return " ".join(tokens)

        for text in texts:
            msg_tokens.append(extract_tokens(text))
        return np.array(msg_tokens)

    @staticmethod
    def get_length(texts):
        """_summary_

        Args:
            texts (_type_): _description_

        Returns:
            _type_: _description_
        """
        return np.array([len(text) for text in texts]).reshape(-1, 1)

    def construct_standardizer(self) -> FeatureUnion:
        """
        Constructs a preprocessor for normalization of numerical features.

        Returns:
            Any: A preprocessor object for normalization.
        """

        tfidf_pipeline = Pipeline(
            [
                ("tokenization", FunctionTransformer(self.get_tokens, validate=False)),
                ("bag_of_words", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
            ]
        )

        # Pipeline for scaling the length feature

        length_pipeline = Pipeline(
            [
                (
                    "length_extractor",
                    FunctionTransformer(self.get_length, validate=False),
                ),
                ("min_max_scaler", MinMaxScaler()),
            ]
        )

        # Combine the pipelines
        standardizer = FeatureUnion(
            [("tfidf_transform", tfidf_pipeline), ("length_transform", length_pipeline)]
        )
        logger.info("Standardizer object created successfully")
        return standardizer

    def transform_train_test_data(self) -> tuple:
        """
        Transforms the train and test data using the constructed preprocessor.

        Returns:
            tuple: A tuple containing two sparse matrix, one for transformed
            training data and one for transformed test data.
        """
        try:
            # Read train and test data files
            train_df = pd.read_csv(self.train_data_path)
            test_df = pd.read_csv(self.test_data_path)

            # Get features and target
            msg_col = self.column_names.text_col
            label_col = self.column_names.label_col

            # Spilt train & test data in-terms of features and targets
            x_train, y_train = train_df[msg_col], train_df[label_col]
            x_test, y_test = test_df[msg_col], test_df[label_col]

            # Log the shapes
            logger.info("The shape of X_train: %s", x_train.shape)
            logger.info("The shape of y_train: %s", y_train.shape)
            logger.info("The shape of X_test: %s", x_test.shape)
            logger.info("The shape of y_test: %s", y_test.shape)

            # Get the preprocessor object
            standardizer = self.construct_standardizer()

            # Fit & transform preprocessor with the X_train data
            x_train_normalized = standardizer.fit_transform(x_train)

            # Transform X_test with fitted preprocessor
            x_test_normalized = standardizer.transform(x_test)

            # class ordinal mapping dictionary
            cls_map = self.class_names.to_dict()

            # Convert target labels to numpy arrays
            y_train_ord_arr = np.array([cls_map.get(i) for i in y_train]).reshape(-1, 1)
            y_test_ord_arr = np.array([cls_map.get(i) for i in y_test]).reshape(-1, 1)

            # Create train & test arrays
            train_sparse_matrix = hstack((x_train_normalized, y_train_ord_arr)).tocsr()
            test_sparse_matrix = hstack((x_test_normalized, y_test_ord_arr)).tocsr()

            # Log the shapes
            logger.info(
                "Shape of normalized training sparse matrix: %s",
                train_sparse_matrix.shape,
            )
            logger.info(
                "Shape of normalized test sparse matrix: %s", test_sparse_matrix.shape
            )

            # Save the arrays
            save_npz(self.train_matrix_path, train_sparse_matrix)
            save_npz(self.test_matrix_path, test_sparse_matrix)

            # Create directory if not exist
            create_directories([dirname(self.standardizer_path)])

            # Saving the preprocessor object
            save_as_joblib(self.standardizer_path, standardizer)

            return (train_sparse_matrix, test_sparse_matrix)
        except Exception as e:
            logger.error(CustomException(e))
            raise CustomException(e) from e
