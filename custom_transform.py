from typing import List

import gower
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (KBinsDiscretizer, MinMaxScaler,
                                   OneHotEncoder, QuantileTransformer,
                                   RobustScaler, StandardScaler)

NUMERIC_COLUMNS_DICT = {
        "original":['MR', 'MR.PR', 'MR.Decile', 'TVC', 'TVC.PR', 'TVC.Decile', 'TRC', 'TRC.PR', 'TRC.Decile', 'BAB', 'BAB.PR', 'BAB.Decile', 'EV', 'EV.PR', 'EV.Decile', 'P/B', 'P/B.PR', 'P/B.Decile', 'PSR', 'PSR.PR', 'PSR.Decile', 'ROE', 'ROE.PR', 'ROE.Decile', 'ROA', 'ROA.PR', 'ROA.Decile', 'C/A', 'C/A.PR', 'C/A.Decile', 'D/A', 'D/A.PR', 'D/A.Decile', 'PG', 'PG.PR', 'PG.Decile', 'AG', 'AG.PR', 'AG.Decile'],
        "basic":['MR', 'TRC', 'BAB', 'EV', 'P/B', 'PSR', 'ROA', 'C/A', 'D/A', 'PG', 'AG'],
        "basic-industry-aggregated": ['MR', 'TRC', 'BAB', 'EV', 'P/B', 'PSR', 'ROA', 'C/A', 'D/A', 'PG', 'AG', 'Industry-MR-mean', 'Industry-TRC-mean', 'Industry-BAB-mean', 'Industry-EV-mean', 'Industry-P/B-mean', 'Industry-PSR-mean', 'Industry-ROA-mean', 'Industry-C/A-mean', 'Industry-D/A-mean', 'Industry-PG-mean', 'Industry-AG-mean'],
        "basic-cluster-aggregated": ['MR', 'TRC', 'BAB', 'EV', 'P/B', 'PSR', 'ROA', 'C/A', 'D/A', 'PG', 'AG', 'cluster-MR-mean', 'cluster-TRC-mean', 'cluster-BAB-mean', 'cluster-EV-mean', 'cluster-P/B-mean', 'cluster-PSR-mean', 'cluster-ROA-mean', 'cluster-C/A-mean', 'cluster-D/A-mean', 'cluster-PG-mean', 'cluster-AG-mean'],
        "basic-industry-cluster-aggregated": ['MR', 'TRC', 'BAB', 'EV', 'P/B', 'PSR', 'ROA', 'C/A', 'D/A', 'PG', 'AG', 'Industry-cluster-MR-mean', 'Industry-cluster-TRC-mean', 'Industry-cluster-BAB-mean', 'Industry-cluster-EV-mean', 'Industry-cluster-P/B-mean', 'Industry-cluster-PSR-mean', 'Industry-cluster-ROA-mean', 'Industry-cluster-C/A-mean', 'Industry-cluster-D/A-mean', 'Industry-cluster-PG-mean', 'Industry-cluster-AG-mean', 'Industry-MR-mean', 'Industry-TRC-mean', 'Industry-BAB-mean', 'Industry-EV-mean', 'Industry-P/B-mean', 'Industry-PSR-mean', 'Industry-ROA-mean', 'Industry-C/A-mean', 'Industry-D/A-mean', 'Industry-PG-mean', 'Industry-AG-mean', 'cluster-MR-mean', 'cluster-TRC-mean', 'cluster-BAB-mean', 'cluster-EV-mean', 'cluster-P/B-mean', 'cluster-PSR-mean', 'cluster-ROA-mean', 'cluster-C/A-mean', 'cluster-D/A-mean', 'cluster-PG-mean', 'cluster-AG-mean']
    }

BASIC_COLUMNS = ["Industry","MR","TVC","TRC","BAB","EV","P/B","PSR","ROE","ROA","C/A","D/A","PG","AG"]

def compute_aggregate(data:pd.DataFrame, agg_keys: List[str], numeric_columns:List[str], agg_methods: List[str])->pd.DataFrame:
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

class Clustering(BaseEstimator, TransformerMixin):
    def __init__(self,
                 n_neighbors: int,
                 n_clusters: int):
        self.clustering_model: AgglomerativeClustering
        self.knn: KNeighborsClassifier
        self.encoder: OneHotEncoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown="infrequent_if_exist") 
        self.n_neighbors = n_neighbors
        self.n_clusters = n_clusters

    def fit(self, X: pd.DataFrame, y=None):
        X_ = X.copy()
        if 'Industry' in X.columns:
            encoded_industry = self.encoder.fit_transform(X_[['Industry']])
            _, nc = encoded_industry.shape
            encoded_industry_df = pd.DataFrame(encoded_industry, columns=[f"Industry-encoding-{i}" for i in range(nc)]).astype(int)
            encoded_industry_df.index = X_.index
            X_ = pd.concat([X_, encoded_industry_df], axis=1)
        X_num = X_.select_dtypes("number")
        X_clustered = self.cluster(X_)
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.knn.fit(X_num, X_clustered['cluster'])
        return self

    def cluster(self, X: pd.DataFrame):
        self.clustering_model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage="complete", metric='precomputed')
        distance_matrix = gower.gower_matrix(X)
        clusters_complete = self.clustering_model.fit_predict(distance_matrix)
        X_new = X.copy()
        X_new["cluster"] = np.asanyarray(clusters_complete)
        X_new["cluster"] = X_new["cluster"].astype("category")
        return X_new
    
    def transform(self, X:pd.DataFrame):
        X_encoded = X
        if 'Industry' in X.columns:
            encoded_industry = self.encoder.transform(X[['Industry']])
            _, nc = encoded_industry.shape
            encoded_industry_df = pd.DataFrame(encoded_industry, columns=[f"Industry-encoding-{i}" for i in range(nc)])
            encoded_industry_df.index = X.index
            X_num = X.select_dtypes("number")
            X_encoded = pd.concat([X_num, encoded_industry_df], axis=1)
        X_num = X_encoded.select_dtypes("number")
        X_new = X.copy()
        X_new['cluster'] = self.knn.predict(X_num)
        return X_new
    
    
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self,
                 columns: List[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self #nothing to learn
 
    def transform(self, X:pd.DataFrame):
        return X[self.columns]

class GroupStatsAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 group_cols:List[str],
                 agg_cols:List[str]=['MR', 'TRC', 'BAB', 'EV', 'P/B', 'PSR', 'ROA', 'C/A', 'D/A', 'PG', 'AG'], 
                 agg_funcs:List[str]=['mean']):
        self.agg_cols = agg_cols
        self.agg_funcs = agg_funcs
        self.group_cols = group_cols
        self.stats_df: pd.DataFrame
        

    def fit(self, X:pd.DataFrame, y=None):
        i_stats_df: pd.DataFrame = None
        c_stats_df: pd.DataFrame = None
        ic_stats_df: pd.DataFrame = None
        
        if "Industry" in self.group_cols:
            i_stats_df = compute_aggregate(X, ["Industry"], self.agg_cols, self.agg_funcs).reset_index()
        if "cluster" in self.group_cols:
            c_stats_df = compute_aggregate(X, ["cluster"], self.agg_cols, self.agg_funcs).reset_index()
        if len(self.group_cols)>1:
            ic_stats_df = compute_aggregate(X,  ["Industry","cluster"], self.agg_cols, self.agg_funcs).reset_index()
    
        if len(self.group_cols)>1:
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
        elif "Industry" in self.group_cols:
            self.stats_df = i_stats_df
        else:
            self.stats_df = c_stats_df
        return self
    
    def transform(self, X:pd.DataFrame):
        X = X.merge(self.stats_df, on=self.group_cols, how='left')
        X = X.fillna(X.mean(numeric_only=True))
        return X

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
            self.final_columns = Xt.columns
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
            raise ValueError("I don't know whats the error, Boss!")
        self.final_columns = cols
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

    
    from sklearn.metrics import mean_absolute_error, mean_squared_error
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
        