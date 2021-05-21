import os
from get_data import read_params
import argparse
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest


def high_cor(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:  # we are interested in absolute coefficient value
                col_name = corr_matrix.columns[i]
                col_corr.add(col_name)
    return col_corr


def feature_select(config_path):
    config = read_params(config_path)
    data_path = config["load_data"]["raw_data_csv"]
    df = pd.read_csv(data_path, sep=',', encoding='utf-8')
    target = config["base"]["target"]
    X = df.drop(target, axis=1)
    Y = df[target]
    MI = mutual_info_classif(X, Y)
    MI_series = pd.Series(MI)
    MI_series.index = X.columns
    MI_final = MI_series.sort_values(ascending=False)
    k = config["select_data"]["k"]
    Top_features = SelectKBest(mutual_info_classif, k=k)
    Top_features.fit(X, Y)
    Top_cols = X.columns[Top_features.get_support()]
    X = X[Top_cols]
    corr_value = config["select_data"]["corr_val"]
    corr_features = high_cor(X, corr_value)
    X = X.drop(corr_features, axis=1)
    final_df = pd.concat([X, Y], axis=1)
    filter_data_path = config["filter_data"]["filter_data_csv"]
    final_df.to_csv(filter_data_path, sep=',', index=False, encoding='utf-8')
    return k, corr_value


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    args_parsed = args.parse_args()
    feature_select(config_path=args_parsed.config)
