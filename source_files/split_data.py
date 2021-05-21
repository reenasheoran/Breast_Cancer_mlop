import os
import argparse
from feature_selection import feature_select
import pandas as pd
from sklearn.model_selection import train_test_split
from get_data import read_params


def split_XY(config_path):
    config = read_params(config_path)
    filter_data_path = config["filter_data"]["filter_data_csv"]
    df = pd.read_csv(filter_data_path, sep=',')
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    random_state = config["base"]["random_state"]
    split_ratio = config["split_data"]["test_size"]
    train, test = train_test_split(df, test_size=split_ratio,
                                   random_state=random_state)
    train.to_csv(train_data_path, sep=',', index=False, encoding='utf-8')
    test.to_csv(test_data_path, sep=',', index=False, encoding='utf-8')


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    args_parsed = args.parse_args()
    split_XY(config_path=args_parsed.config)
