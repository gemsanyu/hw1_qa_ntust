import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess-mode",
                        type=str,
                        choices=["p1,p2,p3,p4,p5,p6,p7,p8,p9"])
    parser.add_argument("--use-rfe",
                        type=bool)
    parser.add_argument("--model",
                        type=str)
    args = parser.parse_args()
    return args
    

def run(column_mode: str,
        use_transformation: bool,
        use_rfe: bool,
        n_neighbors:int,
        n_clusters,
        n_features_to_select,
        model_class,
        model_params_dict):
    X_train, y_train = pd.read_csv("X_train.csv").drop(columns=['Unnamed: 0']), pd.read_csv("y_train.csv").drop(columns=['Unnamed: 0'])
    X_test, y_test = pd.read_csv("X_test.csv").drop(columns=['Unnamed: 0']), pd.read_csv("y_test.csv").drop(columns=['Unnamed: 0'])
    print(X_train.dtypes)

if __name__ == "__main__":
    args = prepare_args()
    if args.preprocess_mode == "p1":
        
    
    run()