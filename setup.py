import argparse
from typing import Union

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from models.default_params import (catboost_params, elasticnet_params,
                                   lasso_params, lightgbm_params, mlp_params,
                                   random_forest_params, ridge_params,
                                   svm_params, xgboost_params)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

def prepare_args():
    """
    Parses command-line arguments for model training and preprocessing.

    Returns:
        argparse.Namespace: Parsed arguments including preprocessing mode, model selection,
                            and whether to use RFE for feature selection.
    """
    parser = argparse.ArgumentParser(description="Train regression model with optional preprocessing and RFE")

    # Preprocessing scheme (e.g. different scaling/feature engineering pipelines)
    parser.add_argument(
        "--preprocess-mode",
        type=str,
        choices=["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"],
        help="Select preprocessing pipeline version (p1 to p9)"
    )

    # Whether to use Recursive Feature Elimination (RFE)
    parser.add_argument(
        "--use-rfe",
        action="store_true",
        help="Use RFE (Recursive Feature Elimination) for feature selection"
    )

    # Choice of regression model
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "catboost",
            "xgboost",
            "lightgbm",
            "linear_regression",
            "lasso",
            "ridge",
            "elasticnet",
            "svm",
            "random_forest",
            "mlp"
        ],
        help="Select regression model to train"
    )

    args = parser.parse_args()
    return args


MODEL_TYPES = Union[
    CatBoostRegressor,
    XGBRegressor,
    LGBMRegressor,
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet,
    SVR,
    RandomForestRegressor,
    MLPRegressor
]
def setup_model(model_name:str)->MODEL_TYPES:
    if model_name == "xgboost":
        return XGBRegressor, xgboost_params
    if model_name == "lightgbm":
        return LGBMRegressor, lightgbm_params
    if model_name == "linear_regression":
        return LinearRegression, {}
    if model_name == "lasso":
        return Lasso, lasso_params
    if model_name == "ridge":
        return Ridge, ridge_params
    if model_name == "elasticnet":
        return ElasticNet, elasticnet_params
    if model_name == "svm":
        return SVR, svm_params
    if model_name == "random_forest":
        return RandomForestRegressor, random_forest_params
    if model_name == "mlp":
        return MLPRegressor, mlp_params
    if model_name == "catboost":
        return CatBoostRegressor, catboost_params
    return None, None
    