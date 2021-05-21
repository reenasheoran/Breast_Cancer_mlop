import os
import argparse
import warnings
import sys
from get_data import read_params
import joblib
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def evaluate_accuracy(actual,predicted):
    Accuracy_check= np.round(accuracy_score(actual, predicted),4)
    return Accuracy_check

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    model_dir =  config["model_dir"]
    random_state = config["base"]["random_state"]
    target = config["base"]["target"]
    
    train = pd.read_csv(train_data_path,sep=',')
    test = pd.read_csv(test_data_path, sep=',')

    train_x= train.drop(target,axis=1)
    test_x = test.drop(target,axis=1)

    train_y = train[target]
    test_y = test[target]

    RF = RandomForestClassifier(random_state=random_state)
    RF.fit(train_x,train_y)
    RFPrediction=RF.predict(test_x)
    
    accuracy = evaluate_accuracy(test_y,RFPrediction)
    print("Model accuracy: %s" % accuracy)



if __name__ == "__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    args_parsed=args.parse_args()
    train_and_evaluate(config_path=args_parsed.config)