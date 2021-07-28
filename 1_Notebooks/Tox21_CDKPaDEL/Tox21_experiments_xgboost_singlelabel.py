#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import time

import xgboost as xgb

from Tox21experiment_helper import load_data


import sys
sys.path.append(os.path.join("..","..","0_code"))

if __name__ == "__main__":
	import csv

	labels, all_train_features, test_features, all_train_targets, test_targets, train_id_df, test_id_df, all_train_Fweights, test_Fweights = load_data(
		load_all_features=True,
		load_v1_dataset=False,
		)


	import time
	import dill as pickle

	super_folder = os.path.join("SingleLabelModels")	

	header = ["Model", "Split", "Label", "Metric", "Type", "Score"]
	summary_csv = os.path.join(super_folder,   
		time.strftime("%y%m%d_xgboost_summary.csv", time.localtime()))

	with open(summary_csv, 'w+', newline='', encoding='utf-8') as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerow(header)

	from sklearn.model_selection import ParameterGrid
	from sklearn.utils.class_weight import compute_sample_weight
	from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

	param_grid= {
		'gamma': [0,10], 
		'use_label_encoder': [False],
		'max_depth': [6],
		'subsample': [1], 
		'colsample_bytree': [1], #[0.7],
		'nrounds': [100000],
		'min_child_weight': [1], 
		'max_delta_step': [0], #Consider changing to 1-10 for controling the update 
		'eta': [0.3], #[0.05],
	}

	for label_idx, label in enumerate(labels):
		valid_train = np.where(~np.isnan(all_train_targets[:, label_idx]))[0]
		valid_test = np.where(~np.isnan(test_targets[:, label_idx]))[0]

		train_sample_weight = compute_sample_weight("balanced", all_train_targets[valid_train, label_idx])
		for params in ParameterGrid(param_grid):
			clf = xgb.XGBClassifier(
				**params
				)
			clf.fit(
				X=all_train_features[valid_train], 
				y=all_train_targets[valid_train, label_idx],
				sample_weight=train_sample_weight,
				eval_set=[(all_train_features[valid_train], all_train_targets[valid_train, label_idx]),
						(test_features[valid_test], test_targets[valid_test, label_idx])],	
				eval_metric= "auc",							
				early_stopping_rounds=50, 
				)

			test_predict=clf.predict_proba(test_features[valid_test])[:, 1]
			auc = roc_auc_score(test_targets[valid_test, label_idx], test_predict)
			# test_predict_class = clf.predict(test_features[valid_test])			
			# f1 = f1_score(test_targets[valid_test, label_idx], test_predict_class)
			# acc = accuracy_score(test_targets[valid_test, label_idx], test_predict_class)

			with open(summary_csv, 'a', newline='', encoding='utf-8') as f:
				writer = csv.writer(f, delimiter=',')
				writer.writerow(["xgboost", 0, label, "AUC", "Test", auc])			

			del clf

	sys.exit()