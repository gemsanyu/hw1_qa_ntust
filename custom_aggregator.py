from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


def compute_aggregate(data, agg_keys, numeric_columns, agg_methods):
    stats_df_list = []
    cat_colums_str = "-".join(agg_keys)
    for numeric_col in numeric_columns:
        stats_df = data.groupby(agg_keys)[numeric_col].agg(agg_methods).rename(
            columns={agg_method:f"{cat_colums_str}-{numeric_col}-{agg_method}" for agg_method in agg_methods}
        )
        stats_df_list += [stats_df]
    stats_df = stats_df_list[0]
    for stats_df_ in stats_df_list[1:]:
        stats_df = stats_df.merge(stats_df_, on=agg_keys, how='left')
    return stats_df

def get_stats_df(train_data):
    cat_columns = ["Industry","cluster"]
    numeric_columns = ['MR', 'TRC', 'BAB', 'EV', 'P/B', 'PSR', 'ROA', 'C/A', 'D/A', 'PG', 'AG']
    agg_methods = ["mean","median","max"]
    


class GroupStatsAggregator(BaseEstimator, TransformerMixin):
    def __init__(self,
                 group_cols=["Industry","cluster"], 
                 agg_cols=['MR', 'TRC', 'BAB', 'EV', 'P/B', 'PSR', 'ROA', 'C/A', 'D/A', 'PG', 'AG'], 
                 agg_funcs=['mean']):
        self.group_cols = group_cols
        self.agg_cols = agg_cols
        self.agg_funcs = agg_funcs
        self.stats_df: pd.DataFrame
        
    def fit(self, X, y=None):
        i_stats_df = compute_aggregate(X, ["Industry"], self.agg_cols, self.agg_funcs).reset_index()
        c_stats_df = compute_aggregate(X, ["cluster"], self.agg_cols, self.agg_funcs).reset_index()
        ic_stats_df = compute_aggregate(X,  self.agg_cols, self.agg_cols, self.agg_funcs).reset_index()
        # final_stats_df = i_stats_df.merge(c_stats_df, how='outer')
        print(ic_stats_df.head())
        print(i_stats_df["Industry"])
        final_stats_df = ic_stats_df
        for col in i_stats_df.columns:
            if col=="Industry":
                continue
            final_stats_df[col] = final_stats_df["Industry"].map(i_stats_df.set_index('Industry')[col])
        for col in c_stats_df.columns:
            if col=="cluster":
                continue
            final_stats_df[col] = final_stats_df["cluster"].map(c_stats_df.set_index('cluster')[col])
        self.stats_df = final_stats_df
        return self
    
    def transform(self, X):
        X = X.merge(self.stats_df, on=self.group_cols, how='left')
        return X