svm_params = {
    'kernel': 'rbf',           # 'rbf' is most common; try 'linear' for speed
    'C': 1.0,                  # Regularization (higher = less regularization)
    'epsilon': 0.1,            # No penalty within ±epsilon of true value
    'gamma': 'scale',          # Auto scaling based on input features
}

random_forest_params = {
    'n_estimators': 100,       # More trees = more stable but slower
    'max_depth': None,         # Let trees expand fully
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',    # √n_features is common for regression
    'bootstrap': True,
    'random_state': 42,
    'n_jobs': -1               # Use all CPU cores
}

catboost_params = {
    'iterations': 1000,              # Number of boosting iterations (trees). Can be adjusted based on convergence.
    'learning_rate': 0.05,           # The step size in the gradient descent. A smaller value can give better results, but might require more iterations.
    'depth': 6,                      # Depth of the tree. A deeper tree can model more complex data but might lead to overfitting.
    'l2_leaf_reg': 3,                # L2 regularization coefficient (penalizes large weights).
    'model_size_reg': 0.5,           # Regularization term that helps control model size.
    'border_count': 128,             # The number of splits (borders) for continuous features. Higher values can improve accuracy but increase computation time.
    'bagging_temperature': 0.8,      # Controls the amount of randomness in the bagging process. Higher values make the model more deterministic.
    'random_strength': 1,            # Controls the randomness in the model (helpful for overfitting).
    'cat_features': [],              # List of categorical feature indices. You should list categorical features here.
    'thread_count': -1,              # Number of threads to use for training. Use `-1` to utilize all available threads.
    'task_type': 'CPU',              # 'CPU' for CPU training, 'GPU' for GPU training. Use GPU if available.
    'verbose': 0,                  # Frequency of logging the progress during training (default: 100). 
    'early_stopping_rounds': 50,     # Early stopping rounds. If no improvement in the validation score, training will stop after this many iterations.
    'eval_metric': 'RMSE',  
    "allow_writing_files":False         # Evaluation metric to use (can also be MAE, LogLoss, etc.).
}

mlp_params = {
    'hidden_layer_sizes': (64,32,16,),  # input->64->32->16->1
    'activation': 'relu',          # or 'tanh'
    'solver': 'adam',              # Good default
    'alpha': 0.0001,               # L2 penalty
    'batch_size': 'auto',
    'learning_rate': 'adaptive',   # Keeps learning rate stable
    'max_iter': 500,
    'early_stopping': True,
    'random_state': 42
}

elasticnet_params = {
    'alpha': 0.1,              # Overall regularization strength
    'l1_ratio': 0.5,           # 0 = Ridge, 1 = Lasso; 0.5 = balanced
    'fit_intercept': True,
    'max_iter': 10000,
    'tol': 1e-4,
    'selection': 'cyclic',     # or 'random'
    'random_state': 42
}

ridge_params = {
    'alpha': 1.0,              # Regularization strength
    'fit_intercept': True,
    'max_iter': 10000,
    'tol': 1e-4,
    'solver': 'auto',          # or 'saga' for large datasets
    'random_state': 42
}

lasso_params = {
    'alpha': 0.01,             
    'max_iter': 5000,          
    'tol': 1e-4,
    'selection': 'cyclic',
    'random_state': 42
}

xgboost_params = {'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'verbosity': 0}

lightgbm_params = {
    'boosting_type': 'gbdt',
    'n_estimators': 1000,             # more trees, early stopping will help prevent overfitting
    'learning_rate': 0.01,            # small LR for better generalization
    'num_leaves': 31,                 # can be increased (e.g., 50 or 100) if data is large
    'max_depth': -1,                  # -1 means no limit
    'min_child_samples': 10,          # less than default to let model learn small patterns
    'subsample': 0.8,                 # for bagging
    'subsample_freq': 1,              # bagging every iteration
    'colsample_bytree': 0.8,          # feature subsampling
    'reg_alpha': 1.0,                 # L1 regularization
    'reg_lambda': 1.0,                # L2 regularization
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1
}
