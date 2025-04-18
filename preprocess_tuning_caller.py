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

def call_tuning(preprocess_mode, use_rfe, model):
    cmd_args = ["python",
         "tuning.py",
         "--preprocess-mode",
         preprocess_mode,
         "--model",
         model]
    if use_rfe:
        cmd_args.append("--use-rfe")
    run(cmd_args)

if __name__ == "__main__":
    for model in MODEL_CHOICES:
        for preprocess_mode in PREPROCESS_MODES:
            for use_rfe in [False, True]:
                print(model, preprocess_mode, use_rfe)
                call_tuning(preprocess_mode, use_rfe, model)