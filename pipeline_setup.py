from custom_transform import (BASIC_COLUMNS, NUMERIC_COLUMNS_DICT, Clustering,
                              ColumnSelector, DataFrameWrapper,
                              GroupStatsAggregator)
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import RFE
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (KBinsDiscretizer, MinMaxScaler,
                                   QuantileTransformer, RobustScaler,
                                   StandardScaler)
from xgboost import XGBRegressor


def get_xgboost_default_params():
    xgb_params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'objective': 'reg:squarederror',
        'random_state': 42,
        'verbosity': 0
    }
    return xgb_params
    
# def get_cnn_default_params():
    

def prepare_model_pipeline(column_mode: str,
                            use_transformation: bool,
                            use_rfe: bool,
                            n_neighbors,
                            n_clusters,
                            n_features_to_select,
                            model_class,
                            model_params_dict):
    steps = []
    if column_mode != "original":
        steps.append(("basic_column_selector",ColumnSelector(BASIC_COLUMNS)))
    
    if column_mode == "basic-industry-aggregated":
        steps.append(("gsa",GroupStatsAggregator(group_cols=["Industry"])))
    elif column_mode == "basic-cluster-aggregated":
        steps.append(("clustering", Clustering(n_neighbors=n_neighbors, n_clusters=n_clusters)))
        steps.append(("gsa",GroupStatsAggregator(group_cols=["cluster"])))
    elif column_mode == "basic-industry-cluster-aggregated":
        steps.append(("clustering", Clustering(n_neighbors=n_neighbors, n_clusters=n_clusters)))
        steps.append(("gsa",GroupStatsAggregator(group_cols=["Industry","cluster"])))
    
    if use_transformation:
        numeric_columns = NUMERIC_COLUMNS_DICT[column_mode]
        transforms = [
        ('mms', DataFrameWrapper(MinMaxScaler(), columns=numeric_columns)),
        ('ss', DataFrameWrapper(StandardScaler(), columns=numeric_columns)),
        ('rs', DataFrameWrapper(RobustScaler(), columns=numeric_columns)),
        ('qt', DataFrameWrapper(QuantileTransformer(n_quantiles=100, output_distribution='normal'), columns=numeric_columns)),
        ('kbd', DataFrameWrapper(KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform'), columns=numeric_columns)),
        ('svd', DataFrameWrapper(TruncatedSVD(n_components=7), columns=numeric_columns)),
        ]
        fu = FeatureUnion(transforms).set_output(transform="pandas")
        fu = DataFrameWrapper(fu)
        preprocessor = ColumnTransformer([
            ('num', fu, numeric_columns)
        ], remainder="passthrough")
        preprocessor.set_output(transform="pandas")
        wrapped_preprocessor = DataFrameWrapper(preprocessor)
        steps.append(("transformation",wrapped_preprocessor))
    
    if use_rfe:
        rfe_estimator = model_class(**model_params_dict)
        rfe = RFE(estimator=rfe_estimator, n_features_to_select=n_features_to_select)
        steps.append(("rfe",rfe))
    steps.append(("regressor",model_class(**model_params_dict)))
    model_pipeline = Pipeline(steps)
    return model_pipeline