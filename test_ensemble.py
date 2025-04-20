import json
import pathlib
import pickle

import numpy as np
import pandas as pd

from setup import prepare_args
from sklearn.metrics import mean_absolute_error, mean_squared_error

    

def test(model):
    X_train, y_train = pd.read_csv("X_train.csv").drop(columns=['Unnamed: 0']), pd.read_csv("y_train.csv").drop(columns=['Unnamed: 0'])
    X_test, y_test = pd.read_csv("X_test.csv").drop(columns=['Unnamed: 0']), pd.read_csv("y_test.csv").drop(columns=['Unnamed: 0'])
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    scores = {'MSE': mse,'RMSE': rmse,'MAE': mae}
    return scores
    
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
    trained_model = None
    save_path = pathlib.Path()/"models"/f"{args.model}_model.pkl"
    with open(save_path.absolute(), "rb") as f:
        trained_model = pickle.load(f)
    score_dict = test(trained_model)
    test_result_filepath = pathlib.Path()/"test_results.csv"
    
    with open(test_result_filepath.absolute(), "a+") as result_file:
        result_file.write(f"{args.model},{score_dict['RMSE']},{score_dict['MSE']},{score_dict['MAE']}\n")
    