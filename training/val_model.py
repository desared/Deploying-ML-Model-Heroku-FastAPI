"""
Module to validate the model.
"""
import pandas as pd
from joblib import load
from .modelling.model import compute_score_per_slice


def val_model(test_df: pd.DataFrame, cat_features: list, root_dir: str):
    """
    Validate the trained model
    Parameters
    ----------
    test_df : pd.DataFrame
        Dataframe containing the test features and label. Columns in
        `categorical_features`
    cat_features: list[str]
        List containing the names of the categorical features (default=[])
    root_dir: str
        Path to local model store

    Returns
    -------

    """
    # load model and encoder
    trained_model = load(f"{root_dir}/model/model.joblib")
    encoder = load(f"{root_dir}/model/encoder.joblib")
    lb = load(f"{root_dir}/model/lb.joblib")

    compute_score_per_slice(
        trained_model,
        test_df,
        encoder,
        lb,
        cat_features,
        root_dir
    )
