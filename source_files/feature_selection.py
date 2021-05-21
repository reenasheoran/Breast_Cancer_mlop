import os
from get_data import read_params
import argparse
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

def high_cor(dataset,threshold):
    col_corr=set()
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>threshold: # we are interested in absolute coefficient value
                col_name=corr_matrix.columns[i]
                col_corr.add(col_name)
    return col_corr 

# X.corr()

# corr.add(col_name)
#     return col_corr 

# 
# set(corr_features)
# X=X.drop(corr_features,axis=1)

def feature_select(config_path):
    config = read_params(config_path)
    data_path = config["load_data"]["raw_data_csv"]
    df = pd.read_csv(data_path,sep=',',encoding ='utf-8')
    target = config["base"]["target"]
    X=df.drop(target,axis=1)
    Y=df[target]
    MI=mutual_info_classif(X,Y)
    MI_series=pd.Series(MI)
    MI_series.index=X.columns
    MI_final=MI_series.sort_values(ascending=False)
    Top15 = SelectKBest(mutual_info_classif,k=15)
    Top15.fit(X,Y)
    Top15_cols=X.columns[Top15.get_support()]
    X=X[Top15_cols]
    corr_value=config["select_data"]["corr_val"]
    corr_features=high_cor(X,corr_value)
    X=X.drop(corr_features,axis=1)
    return X, Y
    
if __name__ == "__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    args_parsed=args.parse_args()
    feature_select(config_path=args_parsed.config)