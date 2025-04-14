from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import gower

from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer

def compute_aggregate(data, agg_keys, numeric_columns, agg_methods):
    stats_df_list = []
    cat_colums_str = "-".join(agg_keys)
    for numeric_col in numeric_columns:
        stats_df = data.groupby(agg_keys, observed=False)[numeric_col].agg(agg_methods).rename(
            columns={agg_method:f"{cat_colums_str}-{numeric_col}-{agg_method}" for agg_method in agg_methods}
        )
        stats_df_list += [stats_df]
    stats_df = stats_df_list[0]
    for stats_df_ in stats_df_list[1:]:
        stats_df = stats_df.merge(stats_df_, on=agg_keys, how='left')
    return stats_df

class GroupStatsAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 n_clusters=5,
                 n_neighbors=1,
                 agg_cols=['MR', 'TRC', 'BAB', 'EV', 'P/B', 'PSR', 'ROA', 'C/A', 'D/A', 'PG', 'AG'], 
                 agg_funcs=['mean']):
        self.agg_cols = agg_cols
        self.agg_funcs = agg_funcs
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.stats_df: pd.DataFrame
        self.X_train: pd.DataFrame
        self.clustering_model: AgglomerativeClustering
        self.knn: KNeighborsClassifier
        self.encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown="infrequent_if_exist") 

    def prepare_knn(self):
        if 'Industry' in self.X_train.columns:
            encoded_industry = self.encoder.fit_transform(self.X_train[['Industry']])
            _, nc = encoded_industry.shape
            encoded_industry_df = pd.DataFrame(encoded_industry, columns=[f"Industry-encoding-{i}" for i in range(nc)]).astype(int)
            encoded_industry_df.index = self.X_train.index
            self.X_train = pd.concat([self.X_train, encoded_industry_df], axis=1)
        X_num = self.X_train.select_dtypes("number")
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.knn.fit(X_num, self.X_train['cluster'])
    
    def cluster(self, X):
        self.clustering_model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage="complete", metric='precomputed')
        distance_matrix = gower.gower_matrix(X)
        clusters_complete = self.clustering_model.fit_predict(distance_matrix)
        X["cluster"] = np.asanyarray(clusters_complete)
        X["cluster"] = X["cluster"].astype("category")
        self.X_train = X.copy()
        self.prepare_knn()
        return X
    
    def fit(self, X, y=None):
        X = self.cluster(X)
        i_stats_df = compute_aggregate(X, ["Industry"], self.agg_cols, self.agg_funcs).reset_index()
        c_stats_df = compute_aggregate(X, ["cluster"], self.agg_cols, self.agg_funcs).reset_index()
        ic_stats_df = compute_aggregate(X,  ["Industry","cluster"], self.agg_cols, self.agg_funcs).reset_index()
        final_stats_df = ic_stats_df
        for col in i_stats_df.columns:
            if col in ["Industry","cluster"]:
                continue
            final_stats_df[col] = final_stats_df["Industry"].map(i_stats_df.set_index('Industry')[col]).astype(float)
            final_stats_df[col] = final_stats_df[col].fillna(final_stats_df[col].mean())
        for col in c_stats_df.columns:
            if col in ["Industry","cluster"]:
                continue
            final_stats_df[col] = final_stats_df["cluster"].map(c_stats_df.set_index('cluster')[col]).astype(float)
            final_stats_df[col] = final_stats_df[col].fillna(final_stats_df[col].mean())
        self.stats_df = final_stats_df
        return self
    
    def transform(self, X):
        if 'Industry' in X.columns:
            encoded_industry = self.encoder.transform(X[['Industry']])
            _, nc = encoded_industry.shape
            encoded_industry_df = pd.DataFrame(encoded_industry, columns=[f"Industry-encoding-{i}" for i in range(nc)])
            encoded_industry_df.index = X.index
            X_num = X.select_dtypes("number")
            X_encoded = pd.concat([X_num, encoded_industry_df], axis=1)
        X['cluster'] = self.knn.predict(X_encoded)
        X = X.merge(self.stats_df, on=["Industry","cluster"], how='left')
        X = X.fillna(X.mean(numeric_only=True))
        return X
    
    
class CatBoostWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kparams):
        self.kparams = kparams
        self.model: CatBoostRegressor = CatBoostRegressor(**self.kparams)
        # self.cat_columns = ["remainder__Industry", "remainder__cluster", "Industry","cluster"]

    def fit(self, X, y):
        # current_cat_columns = []
        # for col in self.cat_columns:
        #     if col in X.columns:
        #         current_cat_columns.append(col)
        #         X[col] = X[col].astype("category")
        self.model = CatBoostRegressor(**self.kparams)
        self.model.fit(X, y, verbose=0)
        return self
    
    def _fit(self, X, y, **fit_params):
        current_cat_columns = []
        for col in self.cat_columns:
            if col in X.columns:
                current_cat_columns.append(col)
                X[col] = X[col].astype("category")
        self.model = CatBoostRegressor(**self.kparams)
        self.model.fit(X, y, cat_features=current_cat_columns, verbose=0)
        return self
    
    

    def predict(self, X):
        return self.model.predict(X)

class DataFrameWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, transformer, columns=None):
        self.transformer = transformer
        self.columns = columns
    
    def set_output(self, transform=None):
        # Just a stub to allow set_output() calls to pass
        return self

    def fit(self, X, y=None):
        self.columns_ = self.columns or X.columns.tolist()
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        Xt = self.transformer.transform(X)
        if isinstance(Xt, pd.DataFrame):
            return Xt  # already good
        if hasattr(self.transformer, 'get_feature_names_out'):
            try:
                cols = self.transformer.get_feature_names_out(self.columns_)
            except AttributeError as e:
                cols = self.columns_
            cols = [f"{col}-{self.transformer.__class__.__name__}" for col in cols]
        elif Xt.shape[1] == len(self.columns_):
            cols = self.columns_
            cols = [f"{col}-{self.transformer.__class__.__name__}" for col in cols]
        else:
            raise ValueError("ASDSA")
        return pd.DataFrame(Xt, columns=cols, index=X.index)

    
if __name__ == "__main__":

    data = pd.read_csv("final.csv", encoding="latin-1")
    data["Industry"] = data["Industry"].astype("str")
    data = data.drop(columns=["Unnamed: 0","cluster"])
    
    numeric_columns = ['MR', 'TRC', 'BAB', 'EV', 'P/B', 'PSR', 'ROA', 'C/A', 'D/A', 'PG', 'AG', 'Industry-cluster-MR-mean', 'Industry-cluster-TRC-mean', 'Industry-cluster-BAB-mean', 'Industry-cluster-EV-mean', 'Industry-cluster-P/B-mean', 'Industry-cluster-PSR-mean', 'Industry-cluster-ROA-mean', 'Industry-cluster-C/A-mean', 'Industry-cluster-D/A-mean', 'Industry-cluster-PG-mean', 'Industry-cluster-AG-mean', 'Industry-MR-mean', 'Industry-TRC-mean', 'Industry-BAB-mean', 'Industry-EV-mean', 'Industry-P/B-mean', 'Industry-PSR-mean', 'Industry-ROA-mean', 'Industry-C/A-mean', 'Industry-D/A-mean', 'Industry-PG-mean', 'Industry-AG-mean', 'cluster-MR-mean', 'cluster-TRC-mean', 'cluster-BAB-mean', 'cluster-EV-mean', 'cluster-P/B-mean', 'cluster-PSR-mean', 'cluster-ROA-mean', 'cluster-C/A-mean', 'cluster-D/A-mean', 'cluster-PG-mean', 'cluster-AG-mean']
    cat_columns = ['Industry', 'cluster']
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
        ('num', fu, numeric_columns),
    ], remainder="passthrough")
    preprocessor.set_output(transform="pandas")
    wrapped_preprocessor = DataFrameWrapper(preprocessor)
    steps = []
    steps.append(("gsa",GroupStatsAggregator()))
    steps.append(("preprocess",wrapped_preprocessor))
    steps.append(("regressor",CatBoostWrapper(iterations=100,depth=5,learning_rate=0.1,verbose=0)))
    model = Pipeline(steps)

    
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # Loop over each fold for cross-validation
    fold_metrics = []
    fold_counter = 1
    x = data.drop(columns=["Yt.1M"])
    y = data["Yt.1M"]
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1
    }
    for train_index, test_index in kf.split(x):
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = Pipeline(steps)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        fold_metrics.append({
        'Fold': fold_counter,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae
        })
        print(f"Fold {fold_counter} -- MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        