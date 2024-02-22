"""
This module contains a class for training a machine learning model using
ElasticNet regression. The class reads in configuration files, prepares the model
with specified hyperparameters, trains the model on a given dataset, and
saves the trained model.
"""

from os.path import dirname, normpath

from scipy.sparse import load_npz
from sklearn.naive_bayes import MultinomialNB

from src.constants import CONFIGS
from src.exception import CustomException
from src.logger import logger
from src.utils.basic_utils import create_directories, read_yaml, save_as_joblib


class ModelTrainer:
    """
    A class used to train a machine learning model using linear regression.
    """

    def __init__(self):
        # Read the configuration files
        self.configs = read_yaml(CONFIGS).model_trainer

        # Input file path
        self.train_matrix_path = normpath(self.configs.train_matrix_path)

        # Output file path
        self.model_path = normpath(self.configs.model_path)

    def train_model(self) -> MultinomialNB:
        """
        summary
        """
        try:
            # Load the training and test set array
            train_array = load_npz(self.train_matrix_path)

            # Split train_array into features and target
            x_train, y_train = (
                train_array[:, :-1],
                train_array[:, -1].toarray().squeeze(),
            )

            # Log the train shapes
            logger.info("The shape of x_train: %s", x_train.shape)
            logger.info("The shape of y_train: %s", y_train.shape)

            # Prepare the model
            nb_classifier = MultinomialNB()
            logger.info("Multinomial Naive Bayes model object initiated")

            # Fit the model on training dataset
            nb_classifier.fit(x_train, y_train)
            logger.info("Multinomial Naive Bayes model fitted on training set")

            # Create directory if not exist
            create_directories([dirname(self.model_path)])

            # Saving the preprocessor object
            save_as_joblib(self.model_path, nb_classifier)

            return nb_classifier
        except Exception as e:
            logger.error(CustomException(e))
            raise CustomException(e) from e
