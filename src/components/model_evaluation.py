"""
wip
"""

from os.path import dirname, join, normpath

import numpy as np
from scipy.sparse import load_npz

from src.constants import CONFIGS, SCHEMA
from src.exception import CustomException
from src.logger import logger
from src.utils.basic_utils import (
    create_directories,
    load_joblib,
    read_yaml,
    save_as_json,
)
from src.utils.model_utils import (
    classification_metrics,
    detailed_classification_metrics,
    detailed_confusion_matrix,
)


class ModelEvaluation:
    """_summary_"""

    def __init__(self):
        """_summary_"""
        # Read the configuration files
        self.configs = read_yaml(CONFIGS).model_evaluation
        self.schema = read_yaml(SCHEMA)

        # Get the class map & model name
        self.class_map = self.schema.class_names.to_dict()
        self.model_name = self.schema.model_name
        # Input file path
        self.train_matrix_path = normpath(self.configs.train_matrix_path)
        self.test_matrix_path = normpath(self.configs.test_matrix_path)
        self.model_path = normpath(self.configs.model_path)

        # Output file path
        self.scores_path = normpath(self.configs.scores_path)
        self.preds_dir = normpath(self.configs.predictions_dir)

    @staticmethod
    def get_classification_report_dict(y_true, y_pred, classes: list) -> dict:
        """_summary_

        Args:
            y_true (_type_): _description_
            y_pred (_type_): _description_
            classes (list): _description_

        Returns:
            dict: _description_
        """
        metrics = classification_metrics(y_true, y_pred)
        cr_df, avgs_df, _ = detailed_classification_metrics(y_true, y_pred, classes)
        cm_df = detailed_confusion_matrix(y_true, y_pred, classes, normalize=False)
        cm_df_norm = detailed_confusion_matrix(y_true, y_pred, classes, normalize=True)

        cm_dict = cm_df.set_index("class").transpose().to_dict()
        cm_norm_dict = cm_df_norm.set_index("class").transpose().to_dict()
        cr_dict = cr_df.set_index("class").transpose().to_dict()
        avgs_dict = avgs_df.transpose().to_dict()

        return {
            "classification_metrics": metrics,
            "averages": avgs_dict,
            "classification_report": cr_dict,
            "confusion_matrix": cm_dict,
            "confusion_matrix_normalized": cm_norm_dict,
        }

    def get_features_and_labels(self) -> tuple[np.array]:
        """_summary_

        Args:
            self (_type_): _description_

        Raises:
            CustomException: _description_

        Returns:
            _type_: _description_
        """
        try:
            # Load the training & test set array
            train_matrix = load_npz(self.train_matrix_path)
            test_matrix = load_npz(self.test_matrix_path)

            # Split train_array into features and target
            x_train, y_train = (
                train_matrix[:, :-1],
                train_matrix[:, -1].toarray().squeeze(),
            )
            x_test, y_test = (
                test_matrix[:, :-1],
                test_matrix[:, -1].toarray().squeeze(),
            )

            # Log the shapes
            logger.info("The shape of x_train: %s", x_train.shape)
            logger.info("The shape of y_train: %s", y_train.shape)
            logger.info("The shape of x_test: %s", x_test.shape)
            logger.info("The shape of y_test: %s", y_test.shape)
            return (x_train, y_train, x_test, y_test)
        except Exception as e:
            logger.error(CustomException(e))
            raise CustomException(e) from e

    def get_predictions(self) -> tuple[np.array]:
        """_summary_

        Args:
            self (_type_): _description_

        Raises:
            CustomException: _description_

        Returns:
            _type_: _description_
        """

        try:
            # Load the model
            model = load_joblib(self.model_path)

            # load train and test features
            x_train, y_train, x_test, y_test = self.get_features_and_labels()

            # Perform predictions
            y_train_preds = model.predict(x_train)
            y_test_preds = model.predict(x_test)
            logger.info("predictions on training and test data completed")

            # Log the shape
            logger.info("Shape of y_train_preds:%s", {y_train_preds.shape})
            logger.info("Shape of y_test_preds:%s", {y_test_preds.shape})

            return (y_train, y_train_preds, y_test, y_test_preds, model)
        except Exception as e:
            logger.error(CustomException(e))
            raise CustomException(e) from e

    @staticmethod
    def get_class_names(model_class: np.array, class_map: dict) -> list:
        """_summary_

        Args:
            model_class (np.array): _description_
            class_map (dict): _description_

        Returns:
            list: _description_
        """
        cls_names = {v: k for k, v in class_map.items()}
        return [cls_names[cls_id] for cls_id in model_class if cls_id in cls_names]

    def evaluate_model(self) -> dict:
        """_summary_

        Raises:
            CustomException: _description_

        Returns:
            dict: _description_
        """
        try:
            # load train and test predictions
            y_train, y_train_preds, y_test, y_test_preds, model = self.get_predictions()

            # Get Classes
            classes = self.get_class_names(model.classes_, self.class_map)

            # Evaluate model
            train_eval_metrics = self.get_classification_report_dict(
                y_train, y_train_preds, classes
            )
            test_eval_metrics = self.get_classification_report_dict(
                y_test, y_test_preds, classes
            )

            return {
                "train_eval_metrics": train_eval_metrics,
                "test_eval_metrics": test_eval_metrics,
                "y_train_preds": y_train_preds,
                "y_test_preds": y_test_preds,
                "model": model,
            }
        except Exception as e:
            logger.info(CustomException(e))
            raise CustomException(e) from e

    def save_evaluation_results(self):
        """_summary_"""
        eval_details = self.evaluate_model()

        train_eval_metrics = eval_details.get("train_eval_metrics")
        test_eval_metrics = eval_details.get("test_eval_metrics")
        y_train_preds = eval_details.get("y_train_preds")
        y_test_preds = eval_details.get("y_test_preds")

        # Create directory to save predictions & model score
        create_directories([self.preds_dir, dirname(self.scores_path)])

        # Save the training predictions
        train_preds_filepath = join(self.preds_dir, "train_preds.npy")
        np.save(train_preds_filepath, y_train_preds)
        logger.info("training predictions saved at: %s", train_preds_filepath)

        # Save the test predictions
        test_preds_filepath = join(self.preds_dir, "test_preds.npy")
        np.save(test_preds_filepath, y_test_preds)
        logger.info("test predictions saved at: %s", test_preds_filepath)

        # Create model scores dict
        scores_dict = {
            "model_name": self.model_name,
            "training_evaluation_metrics": train_eval_metrics,
            "test_evaluation_metrics": test_eval_metrics,
        }

        # Save model scores
        save_as_json(self.scores_path, scores_dict)

        logger.info("Scores recorded in: %s", self.scores_path)
