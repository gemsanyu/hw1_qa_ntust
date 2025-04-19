from subprocess import run

PREPROCESS_MODES = [
                    "p1", 
                    "p2", 
                    "p3", 
                    "p4", 
                    "p5", 
                    "p6", 
                    "p7", 
                    "p8", 
                    "p9"
                    ]
MODEL_CHOICES = [
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
        ]

def call_hyperparam_tuning(model, preprocess_mode, use_rfe):
    cmd_args = ["python",
         "hyperparam_tuning.py",
         "--preprocess-mode",
         preprocess_mode,
         "--model",
         model]
    if use_rfe:
        cmd_args.append("--use-rfe")
    run(cmd_args)

if __name__ == "__main__":
    model_ppmode_list = [
    ("xgboost", "p1", False),
    ("catboost", "p1", False),
    ("elasticnet", "p7", False),
    ("lasso", "p1", True ),
    ("lightgbm", "p9", False),
    ("random_forest", "p5", False),
    ("ridge", "p4", True ),
    ("svm", "p9", False),
    ("mlp", "p9", False )]
    
    for args in model_ppmode_list:
        call_hyperparam_tuning(*args)