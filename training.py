import json
import pathlib
from typing import Optional
import pickle

import numpy as np
import pandas as pd
from pipeline_setup import prepare_model_pipeline

from setup import prepare_args, MODEL_TYPES, setup_model
    

def train(column_mode: str,
        use_transformation: bool,
        use_rfe: bool,
        n_neighbors:Optional[int],
        n_clusters:Optional[int],
        n_features_to_select:Optional[int],
        model_class:MODEL_TYPES,
        model_params_dict:dict)->np.ndarray:
    X_train, y_train = pd.read_csv("X_train.csv").drop(columns=['Unnamed: 0']), pd.read_csv("y_train.csv").drop(columns=['Unnamed: 0'])
    model_pipeline = prepare_model_pipeline(column_mode, use_transformation, use_rfe, n_neighbors,n_clusters,n_features_to_select, model_class, model_params_dict)
    model_pipeline.fit(X_train, y_train)
    return model_pipeline
    
def combine_with_tuned_params(model_name, default_params):
    if model_name == "linear":
        return {}
    tuned_params_filepath = pathlib.Path()/"models"/f"{model_name}_params.json"
    tuned_params = {}
    with open(tuned_params_filepath.absolute(), "r") as json_file:
        tuned_params = json.load(json_file)
    for k,v in tuned_params.items():
        default_params[k] = v
    return default_params


if __name__ == "__main__":
    args = prepare_args()
    if args.model == "mlp" and args.use_rfe:
        raise ValueError("RFE cannot be used on MLP")
    model_class, model_params_dict = setup_model(args.model)
    tuned_params = combine_with_tuned_params(args.model, model_params_dict)
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
    trained_model = train(column_mode,use_transformation,use_rfe,n_neighbors,n_clusters,n_features_to_select,model_class,tuned_params)
    save_path = pathlib.Path()/"models"/f"{args.model}_model.pkl"
    with open(save_path.absolute(),"wb") as f:
        pickle.dump(trained_model, f)