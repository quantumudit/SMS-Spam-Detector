"""
summary
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(y_true, y_pred) -> dict:
    """
    Calculates the classification metrics accuracy, precision, recall, and F1 score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    classification_dict = {
        "Accuracy": round(accuracy, 3),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1 Score": round(f1, 3),
    }
    return classification_dict


def detailed_classification_metrics(y_true, y_pred, classes: list) -> tuple:
    """_summary_

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_
        classes (list): _description_

    Returns:
        tuple: _description_
    """
    cr_dict = classification_report(y_true, y_pred, output_dict=True)
    cr_df = pd.DataFrame(cr_dict).transpose()
    cr_classes = (
        cr_df.iloc[: len(classes)].reset_index().rename(columns={"index": "class"})
    )
    avgs_df = cr_df.iloc[len(classes) + 1 :]
    accuracy = round(cr_dict["accuracy"], 3)
    return (cr_classes, avgs_df, accuracy)


def detailed_confusion_matrix(
    y_true, y_pred, classes: list, normalize: bool = False
) -> pd.DataFrame:
    """_summary_

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_
        classes (list): _description_
        normalize (bool, optional): _description_. Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """
    cm = confusion_matrix(y_true, y_pred)

    cm_data = []

    # Iterate over classes and confusion matrix
    for idx, class_name in enumerate(classes):
        tp = cm[idx][idx]
        fp = sum(cm[row][idx] for row in range(len(cm)) if row != idx)
        fn = sum(cm[idx][col] for col in range(len(cm)) if col != idx)
        tn = sum(sum(row) for j, row in enumerate(cm) if j != idx) - fp
        cm_dict = {
            "class": class_name,
            "TP": tp if normalize is False else tp / (tp + fn),
            "FP": fp if normalize is False else fp / (tn + fp),
            "FN": fn if normalize is False else fn / (tp + fn),
            "TN": tn if normalize is False else tn / (tn + fp),
        }
        cm_data.append(cm_dict)
    # Construct dataframe
    cm_df = pd.DataFrame(cm_data)
    return cm_df


def get_classification_report_df(y_true, y_pred, classes: list) -> pd.DataFrame:
    """_summary_

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_
        classes (list): _description_

    Returns:
        pd.DataFrame: _description_
    """
    cr_df, _, _ = detailed_classification_metrics(y_true, y_pred, classes)
    cm_df = detailed_confusion_matrix(y_true, y_pred, classes, normalize=False)
    cm_df_norm = detailed_confusion_matrix(y_true, y_pred, classes, normalize=True)

    # Add "_norm" to cm_df_norm columns (except the "class" column)
    cm_df_norm.columns = ["class"] + [
        x + "_norm" for x in cm_df_norm.columns if x != "class"
    ]

    # Change the "class" column to "str" type
    for df in [cr_df, cm_df, cm_df_norm]:
        df["class"] = df["class"].astype(str)

    # Merge dataframes
    merged_df = cr_df.merge(cm_df, on="class").merge(cm_df_norm, on="class")
    return merged_df
