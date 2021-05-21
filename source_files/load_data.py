import os
from get_data import read_params, get_data
import argparse
import pandas as pd

def load_save(config_path):
    config = read_params(config_path)
    df = get_data(config_path)
    df=df.drop(['id','Unnamed: 32'],axis=1)
    df['diagnosis']=pd.get_dummies(df['diagnosis'],drop_first=True)
    raw_data_path = config["load_data"]["raw_data_csv"]
    df.to_csv(raw_data_path,sep=',',index=False,encoding='utf-8')
    


if __name__ == "__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    args_parsed=args.parse_args()
    load_save(config_path=args_parsed.config)