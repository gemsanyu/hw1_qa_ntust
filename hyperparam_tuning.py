import pathlib
from typing import Optional
import json

from setup import prepare_args, setup_model, MODEL_TYPES
import numpy as np
import pandas as pd
from pipeline_setup import prepare_model_pipeline
from sklearn.model_selection import RepeatedKFold, cross_val_score
import optuna


def hyperparam_tuning(column_mode: str,
        use_transformation: bool,
        use_rfe: bool,
        n_neighbors:Optional[int],
        n_clusters:Optional[int],
        n_features_to_select:Optional[int],
        model_name: str)->np.ndarray:
    model_class, model_params_dict = setup_model(model_name)
    X_train, y_train = pd.read_csv("X_train.csv").drop(columns=['Unnamed: 0']), pd.read_csv("y_train.csv").drop(columns=['Unnamed: 0'])
    
    def objective(trial):
        params = model_params_dict.copy()
        if model_name == 'svm':
            params['C'] = trial.suggest_float('C', 1e-5, 100, log=True)
            params['epsilon'] = trial.suggest_float('epsilon', 0.01, 0.1)
            params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])

        elif model_name == 'random_forest':
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
            params['max_depth'] = trial.suggest_int('max_depth', 5, 30)
            params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 10)

        elif model_name == 'xgboost':
            params['n_estimators'] = trial.suggest_int('n_estimators', 100, 1000)
            params['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 0.1, log=True)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 10)

        elif model_name == 'lightbgm':
            params['n_estimators'] = trial.suggest_int('n_estimators', 100, 1000)
            params['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 0.1, log=True)
            params['num_leaves'] = trial.suggest_int('num_leaves', 20, 100)

        elif model_name == 'catboost':
            params['iterations'] = trial.suggest_int('iterations', 500, 2000)
            params['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 0.1, log=True)
            params['depth'] = trial.suggest_int('depth', 3, 10)

        elif model_name == 'elasticnet':
            params['alpha'] = trial.suggest_float('alpha', 1e-4, 1.0, log=True)
            params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)

        elif model_name == 'ridge':
            params['alpha'] = trial.suggest_float('alpha', 0.01, 10.0, log=True)

        elif model_name == 'lasso':
            params['alpha'] = trial.suggest_float('alpha', 1e-4, 1.0, log=True)
            
        elif model_name == 'mlp':
            params['hidden_layer_sizes'] = trial.suggest_categorical('hidden_layer_sizes', [(64,32,16), (128,64), (64,64), (100,)])
            params['alpha'] = trial.suggest_loguniform('alpha', 1e-5, 1e-1)  # L2 penalty
            params['learning_rate'] = trial.suggest_categorical('learning_rate', ['constant', 'adaptive'])
        
            
        model_pipeline = prepare_model_pipeline(column_mode, use_transformation, use_rfe, n_neighbors,n_clusters,n_features_to_select, model_class, params)
        cv = RepeatedKFold(n_repeats=3)
        cv_scores = -cross_val_score(model_pipeline, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
        return cv_scores.mean()
    
    # Define the Optuna study
    study = optuna.create_study(direction='minimize')

    # Start the optimization process
    study.optimize(objective, n_trials=100)

    # Print the best parameters
    with open(f"models/{model_name}_params.json", 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print(model_name,":=======================")
    print("Best parameters:", study.best_params)
    print("Best RMSE:", study.best_value)
    

if __name__ == "__main__":
    args = prepare_args()
    if args.model == "mlp" and args.use_rfe:
        raise ValueError("RFE cannot be used on MLP")
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
    hyperparam_tuning(column_mode,use_transformation,use_rfe,n_neighbors,n_clusters,n_features_to_select,args.model)
    