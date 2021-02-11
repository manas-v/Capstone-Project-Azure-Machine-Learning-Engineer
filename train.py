# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.metrics import roc_auc_score

url="https://raw.githubusercontent.com/manas-v/Capstone-Project-Azure-Machine-Learning-Engineer/main/WA_Fn-UseC_-HR-Employee-Attrition.csv"
ds=TabularDatasetFactory.from_delimited_files(path=url)

def clean_data(data):
    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    x_df = x_df.drop(['EmployeeCount', 'Over18', 'StandardHours'], axis = 1) 
    x_df = pd.get_dummies(x_df, drop_first=True)
    y_df = x_df.pop("Attrition")
    return x_df,y_df

x, y = clean_data(ds)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=1)

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--criterion', type=str, default="gini", help="The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.")
    parser.add_argument('--splitter', type=str, default="best", help="The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.")
    parser.add_argument('--max_depth', type=int, help="The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")

    args = parser.parse_args()

    run.log("Attribute selection measure:", np.str(args.criterion))
    run.log("Split Strategy:", np.str(args.splitter))
    run.log("Maximum Depth of a Tree:", np.float(args.max_depth))

    model = DecisionTreeClassifier(criterion=args.criterion, splitter=args.splitter, max_depth=args.max_depth).fit(x_train, y_train)

    AUC_weighted = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1], average="weighted")
    run.log("AUC_weighted", np.float(AUC_weighted))
    
    os.makedirs('./outputs', exist_ok=True)    
    joblib.dump(value=model, filename='./outputs/model.joblib')

if __name__ == '__main__':
    main()
