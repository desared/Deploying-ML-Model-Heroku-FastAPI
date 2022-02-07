"""
Module to train machine learning model.
"""
import pandas as pd
from typing import Tuple

from joblib import dump
from sklearn.model_selection import train_test_split
from .modelling.data import process_data
from .modelling.model import train_model


def get_train_test_data(root_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get data for modelling and testing

    Parameters
    ----------
    root_path : str
        Path of the clean dataset.

    Returns
    -------
    train_df , test_df
    """
    data = pd.read_csv(f"{root_path}/data/clean_census.csv")
    train_df, test_df = train_test_split(data, test_size=0.20)

    return train_df, test_df


def train_save_model(train: pd.DataFrame, cat_features: list, root_path: str):
    """
    Train and save modelling model
    Parameters
    ----------
    root_path: str
        Path to local model store
    train : pd.DataFrame
        Dataframe containing the features and label. Columns in
        `categorical_features`
    cat_features: list[str]
        List containing the names of the categorical features (default=[])

    Returns
    -------

    """
    x_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    # train model
    trained_model = train_model(x_train, y_train)
    # save model
    dump(trained_model, f"{root_path}/model/model.joblib")
    dump(encoder, f"{root_path}/model/encoder.joblib")
    dump(lb, f"{root_path}/model/lb.joblib")
