#!/usr/bin/env python3
# File name: Stacking.py
# Author: XiuHuan Yap
# Contact: yapxiuhuan@gmail.com
"""Train Stacking layer on Acute oral toxicity dataset, from train/test predictions of initial model"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

import time

import dill as pickle

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import r2_score, mean_squared_error

from tensorflow.keras.utils import to_categorical

def get_valid_ind(y):
    ind = np.where(~np.isnan(y.astype(np.float32)))[0]
    return ind

from AOTexperiment_helper import load_data, file_updater
from AOTexperiment_helper import get_cm, get_sn_sp, get_qual_model_score

def get_model_eval(y_train, y_test, train_predict, test_predict):
    cm = get_cm(y_train, train_predict)
    sn_train, sp_train = get_sn_sp(cm)
    cm = get_cm(y_test, test_predict)
    sn_test, sp_test = get_sn_sp(cm)
    model_score = get_qual_model_score(sn_train, sp_train, sn_test, sp_test)
    return model_score, (sn_train, sp_train, sn_test, sp_test)

from sklearn.model_selection import StratifiedKFold, KFold

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_predict_file", 
                    help="path to train_predict file")
parser.add_argument("--test_predict_file", help="path to test_predict file")
parser.add_argument("--model_type", help = "Model type of initial predictions (for naming purposes only)")
parser.add_argument("--train_predict_file2", 
    help="path to second train_predict file (for combined models)",
    default=None)
parser.add_argument("--test_predict_file2", 
    help="path to second test_predict file (for combined models)", default = None
    )
parser.add_argument("--combine_method", 
    help="Choose 'stacked' or 'mean' method for combining initial predictions.", 
    default="mean")
args = parser.parse_args()

if __name__ == "__main__":
    train_predict_file = args.train_predict_file
    test_predict_file = args.test_predict_file
    model_type = args.model_type
    train_predict_file2 = args.train_predict_file2
    test_predict_file2 = args.test_predict_file2
    combine_method = args.combine_method

    train_predict_init = pd.read_csv(train_predict_file, index_col=0).values
    test_predict_init = pd.read_csv(test_predict_file, index_col=0).values

    if train_predict_file2 is not None:
        train_predict_file2 = pd.read_csv(train_predict_file2, index_col=0).values
        test_predict_file2 = pd.read_csv(test_predict_file2, index_col=0).values
        if combine_method=="mean":
            train_predict=np.mean([train_predict_init, train_predict_file2], axis=0)
            test_predict=np.mean([test_predict_init, test_predict_file2], axis=0)
        elif combine_method=="stacked":
            train_predict_init= np.hstack([train_predict_init, train_predict_file2],)
            test_predict_init=np.hstack([test_predict_init, test_predict_file2], )
        else:
            raise ValueError

    data_dict = load_data("data")
    # train_features = data_dict['train_features']
    # test_features = data_dict['test_features']
    train_targets = data_dict['train_targets']
    test_targets = data_dict['test_targets']
    # train_Fweights = data_dict['train_Fweights']
    # test_Fweights = data_dict['test_Fweights']
    labels = data_dict['labels']

    super_folder = "AOT_Stacking_models"
    folder_name = "_".join([time.strftime("%y%m%d", time.localtime()),
                            model_type,
                            "20foldCV"
]
                           )
    summary_csv_path = os.path.join(super_folder,
                                    folder_name+".csv"
                                    )
    header = ["Model", "Label", "Split", "Metric", "Type", "Score"]
    file_updater(summary_csv_path, [header], mode='w+')
    try:
        os.mkdir(os.path.join(super_folder, folder_name))
    except:
        pass
    
    n_splits=20 
    #Inputs: Initial predictions for all labels, real labels for training and evaluation
    #Outputs: model score
    for label_idx, label in enumerate(labels):
        valid_train_ind = get_valid_ind(train_targets[:, label_idx])
        valid_test_ind = get_valid_ind(test_targets[:, label_idx])
        all_y_train = train_targets[valid_train_ind, label_idx].astype(np.float32)
        y_test = test_targets[valid_test_ind, label_idx].astype(np.float32)

        #Load xgb params
        params_path = os.path.join(super_folder, 
            f"210615_dense_params_{label}.ob")
        with open(params_path, 'rb') as f:
          params = pickle.load(f)
        params = params['x']

        learning_rate, n_estimators, max_depth, min_child_weight, gamma, subsample, colsample_bytree = params

        #Binary or Multiclass problem
        if label_idx<2 or label_idx>2:
            objective = "binary:logistic"
            eval_metric = "logloss"
            skf = StratifiedKFold(n_splits=n_splits, random_state=42)
            y_split=(~np.isclose(all_y_train,0.)).astype(int)
            if label_idx>2:
                objective = "multi:softprob"
                eval_metric='mlogloss'
                all_y_train = all_y_train-1
                y_test = y_test-1
            train_sample_weight = compute_sample_weight("balanced", all_y_train)

            for idx, (train_ind, _) in enumerate(skf.split(range(len(y_split)), y_split)):
                X_train = train_predict_init[valid_train_ind[train_ind]]
                y_train = all_y_train[train_ind]
                model = xgb.XGBClassifier(n_estimators=499,
                    use_label_encoder=False,
                    objective = objective, 
                    eval_metric=eval_metric,
                    # learning_rate = learning_rate,
                    # n_estimators=n_estimators, 
                    # max_depth = max_depth, 
                    # min_child_weight = min_child_weight, 
                    # gamma=gamma,
                    # subsample=subsample,
                    # colsample_bytree=colsample_bytree, 
                    n_jobs=-1,
                    )
                model.fit(
                    X_train, 
                    y_train, 
                    sample_weight = train_sample_weight[train_ind]
                    )
                model.save_model(os.path.join(super_folder, folder_name, f"xgb_model_{label}_split{idx:02d}"))

                train_predict_final = model.predict(X_train)
                test_predict_final = model.predict(test_predict_init[valid_test_ind])
                if label_idx<2:
                    r = get_model_eval(y_train, y_test, 
                        train_predict = train_predict_final, 
                        test_predict = test_predict_final)
                else:
                    r = get_model_eval(
                        to_categorical(y_train), 
                        to_categorical(y_test), 
                        to_categorical(train_predict_final), 
                        to_categorical(test_predict_final)
                    )

                model_score, res = r[0], r[1]
                rows = [[model_type, label, idx, "Sensitivity", "Train", res[0]],
                        [model_type, label, idx, "Specificity", "Train", res[1]],
                        [model_type, label, idx, "Sensitivity", "Test", res[2]],
                        [model_type, label, idx, "Specificity", "Test", res[3]]
                ]
                file_updater(summary_csv_path, rows, mode='a')

                row = [model_type, label, idx, "Model Score", "TrainTest", model_score]
                file_updater(summary_csv_path, [row], mode='a')
        #Regression problem
        else:
            all_y_train = np.log10(all_y_train.astype(np.float32))
            y_test = y_test.astype(np.float32)
            ind = np.where(y_test>10000.)[0]
            y_test[ind]=10000.
            y_test = np.log10(y_test)

            kf = KFold(n_splits=n_splits)

            for idx, (train_ind, _) in enumerate(kf.split(range(len(all_y_train)))):
                X_train = train_predict_init[valid_train_ind[train_ind]]
                y_train = all_y_train[train_ind]

                model = xgb.XGBRegressor(
                    n_estimators=499,
                    # learning_rate = learning_rate,
                    # n_estimators=n_estimators, 
                    # max_depth = max_depth, 
                    # min_child_weight = min_child_weight, 
                    # gamma=gamma,
                    # subsample=subsample,
                    # colsample_bytree=colsample_bytree, 
                    objective="reg:squarederror",
                    n_jobs=-1,               
                )
                model.fit(X_train,
                    y_train,
                )
                model.save_model(os.path.join(super_folder, folder_name, f"xgb_{model_type}_{label}_split{idx:02d}"))

                #Predict and eval
                train_predict_final = model.predict(X_train)
                test_predict_final = model.predict(test_predict_init[valid_test_ind])

                r2_train = r2_score(y_train, train_predict_final)
                r2_test = r2_score(y_test, test_predict_final)

                rows = [[model_type, label, idx, "Rsquared", "Train", r2_train],
                        [model_type, label, idx, "Rsquared", "Test", r2_test]
                ]
                file_updater(summary_csv_path, rows, mode='a')
                model_score = (0.3*r2_train) + (0.45*r2_test) +0.25*(np.abs(r2_train-r2_test))
            
                row = [model_type, label, idx, "Model Score", "TrainTest", model_score]
                file_updater(summary_csv_path, [row], mode='a')
    sys.exit()


