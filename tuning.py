import argparse
from typing import Optional, Union

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from models.default_params import (catboost_params, elasticnet_params,
                                   lasso_params, lightgbm_params, mlp_params,
                                   random_forest_params, ridge_params,
                                   svm_params, xgboost_params)
from pipeline_setup import prepare_model_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import RepeatedKFold, cross_val_score
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
    

def run_cv(column_mode: str,
        use_transformation: bool,
        use_rfe: bool,
        n_neighbors:Optional[int],
        n_clusters:Optional[int],
        n_features_to_select:Optional[int],
        model_class:MODEL_TYPES,
        model_params_dict:dict)->np.ndarray:
    X_train, y_train = pd.read_csv("X_train.csv").drop(columns=['Unnamed: 0']), pd.read_csv("y_train.csv").drop(columns=['Unnamed: 0'])
    model_pipeline = prepare_model_pipeline(column_mode, use_transformation, use_rfe, n_neighbors,n_clusters,n_features_to_select, model_class, model_params_dict)
    cv = RepeatedKFold(n_repeats=3)
    cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    return cv_scores


if __name__ == "__main__":
    args = prepare_args()
    if args.model == "mlp" and args.use_rfe:
        raise ValueError("RFE cannot be used on MLP")
    model_class, model_params_dict = setup_model(args.model)
    column_mode="original"
    use_transformation=False
    use_rfe=args.use_rfe
    default_n_neighbors = 3
    default_n_clusters = 8
    n_neighbors=None
    n_clusters=None
    n_features_to_select= 10 if args.use_rfe else None
    if args.preprocess_mode == "p1":
        column_mode="original"
        use_transformation=False
        n_neighbors=None
        n_clusters=None
    if args.preprocess_mode == "p2":
        column_mode="basic"
        use_transformation=False
        n_neighbors=None
        n_clusters=None
    if args.preprocess_mode == "p3":
        column_mode="basic"
        use_transformation=True
        n_neighbors=None
        n_clusters=None
    if args.preprocess_mode == "p4":
        column_mode="basic-industry-aggregated"
        use_transformation=False
        n_neighbors=None
        n_clusters=None
    if args.preprocess_mode == "p5":
        column_mode="basic-cluster-aggregated"
        use_transformation=False
        n_neighbors=default_n_neighbors
        n_clusters=default_n_clusters
    if args.preprocess_mode == "p6":
        column_mode="basic-industry-cluster-aggregated"
        use_transformation=False
        n_neighbors=default_n_neighbors
        n_clusters=default_n_clusters
    if args.preprocess_mode == "p7":
        column_mode="basic-industry-aggregated"
        use_transformation=True
        n_neighbors=None
        n_clusters=None
    if args.preprocess_mode == "p8":
        column_mode="basic-cluster-aggregated"
        use_transformation=True
        n_neighbors=default_n_neighbors
        n_clusters=default_n_clusters
    if args.preprocess_mode == "p9":
        column_mode="basic-industry-aggregated"
        use_transformation=True
        n_neighbors=default_n_neighbors
        n_clusters=default_n_clusters
    cv_scores = run_cv(column_mode,use_transformation,use_rfe,n_neighbors,n_clusters,n_features_to_select,model_class,model_params_dict)
    print(cv_scores, cv_scores.mean())