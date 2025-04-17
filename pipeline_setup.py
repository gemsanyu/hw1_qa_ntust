from custom_aggregator import DataFrameWrapper, GroupStatsAggregator


def prepare_data_preprocessing_pipeline(n_clusters=8,n_neighbors=3,n_features_to_select=12):
    numeric_columns = ['MR', 'TRC', 'BAB', 'EV', 'P/B', 'PSR', 'ROA', 'C/A', 'D/A', 'PG', 'AG', 'Industry-cluster-MR-mean', 'Industry-cluster-TRC-mean', 'Industry-cluster-BAB-mean', 'Industry-cluster-EV-mean', 'Industry-cluster-P/B-mean', 'Industry-cluster-PSR-mean', 'Industry-cluster-ROA-mean', 'Industry-cluster-C/A-mean', 'Industry-cluster-D/A-mean', 'Industry-cluster-PG-mean', 'Industry-cluster-AG-mean', 'Industry-MR-mean', 'Industry-TRC-mean', 'Industry-BAB-mean', 'Industry-EV-mean', 'Industry-P/B-mean', 'Industry-PSR-mean', 'Industry-ROA-mean', 'Industry-C/A-mean', 'Industry-D/A-mean', 'Industry-PG-mean', 'Industry-AG-mean', 'cluster-MR-mean', 'cluster-TRC-mean', 'cluster-BAB-mean', 'cluster-EV-mean', 'cluster-P/B-mean', 'cluster-PSR-mean', 'cluster-ROA-mean', 'cluster-C/A-mean', 'cluster-D/A-mean', 'cluster-PG-mean', 'cluster-AG-mean']
    transforms = [
    ('mms', DataFrameWrapper(MinMaxScaler(), columns=numeric_columns)),
    ('ss', DataFrameWrapper(StandardScaler(), columns=numeric_columns)),
    ('rs', DataFrameWrapper(RobustScaler(), columns=numeric_columns)),
    ('qt', DataFrameWrapper(QuantileTransformer(n_quantiles=100, output_distribution='normal'), columns=numeric_columns)),
    ('kbd', DataFrameWrapper(KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform'), columns=numeric_columns)),
    ]
    fu = FeatureUnion(transforms).set_output(transform="pandas")
    fu = DataFrameWrapper(fu)
    preprocessor = ColumnTransformer([
        ('num', fu, numeric_columns)
    ], remainder="passthrough")
    preprocessor.set_output(transform="pandas")
    wrapped_preprocessor = DataFrameWrapper(preprocessor)
    rfe_estimator = CatBoostRegressor(iterations=iterations, depth=depth, learning_rate=lr, verbose=0)
    rfe = RFE(estimator=rfe_estimator, n_features_to_select=n_features_to_select)

    steps = []
    steps.append(("gsa",GroupStatsAggregator(n_clusters=n_clusters, n_neighbors=n_neighbors)))
    steps.append(("preprocess",wrapped_preprocessor))
    steps.append(("rfe",rfe))
    steps.append(("regressor",CatBoostRegressor(iterations=iterations,depth=depth,learning_rate=lr,verbose=0)))
    model = Pipeline(steps)
    return model