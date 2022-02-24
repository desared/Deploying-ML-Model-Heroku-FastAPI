"""
Module to run the modelling inference.
"""
import numpy as np
from .modelling.data import process_data
from .modelling.model import inference


def run_inference(model, encoder, lb, data: np.array, cat_features: list):
    """
    Load model and run inference.

    Inputs
    ----------
    model: classifier
    encoder: object which encoded the data
    lb: label binarizer object
    data : np.array
        Processed data.
    cat_features: list[str]
        List containing the names of the categorical features (default=[])

    Returns
    -------
    predictions
    """
    X, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        encoder=encoder, lb=lb, training=False)

    pred = inference(model, X)
    prediction = lb.inverse_transform(pred)[0]

    return prediction
