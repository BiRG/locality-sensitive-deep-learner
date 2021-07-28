#!/usr/bin/env python3
# File name: base_rf.py
## This file is used to train single-endpoint Random Forest classifiers against Tox21_extended labels. 

import os, sys
import pandas as pd
import numpy as np
import time
import dill as pickle

from sklearn.preprocessing import StandardScaler

from Tox21_extended_experiment_helper import load_data, get_valid_ind, file_updater, data_scaling, tune_xgbc
from Tox21_extended_experiment_helper import model_eval

if __name__=="__main__":
	data_dict = load_data()
	train_features = data_dict['train_features']
	test_features = data_dict['test_features']
	train_targets = data_dict['train_targets']
	test_targets = data_dict['test_targets']
	# train_Fweights = data_dict['train_Fweights']
	# test_Fweights = data_dict['test_Fweights']
	labels = data_dict['labels']

	feature_scaler = StandardScaler()
	target_scaler = None
	train_features_scaled, test_features_scaled, train_targets_scaled, test_targets_scaled, feature_scaler, target_scaler = data_scaling(
		feature_scaler,
		target_scaler,
		train_features,
		test_features,
		train_targets.values.astype(np.float32),
		test_targets.values.astype(np.float32)
	)

	#Use random forest to optimize
	from skopt import forest_minimize
	from sklearn.utils.class_weight import compute_sample_weight
	from sklearn.metrics import roc_auc_score
	import xgboost as xgb

	# Creating a sample space in which the initial randomic search should be performed
	space = [(1e-3, 1e-1, 'log-uniform'), # learning rate
	          (100, 2000), # n_estimators
	          (1, 10), # max_depth 
	          (1, 6.), # min_child_weight 
	          (0, 0.5), # gamma 
	          (0.5, 1.), # subsample 
	          (0.5, 1.)] # colsample_bytree 

	# Minimization using a random forest with 20 random samples and 50 iterations for Bayesian optimization.
	from functools import partial

	summary_csv_path = os.path.join("base_classifiers", time.strftime("%y%m%d_summary.csv", time.localtime()))
	header = ["Model", "Label", "Metric", "Type", "Score"]	
	file_updater(summary_csv_path, [header], mode='w')

	for label_idx, label in enumerate(labels):
		# if label_idx<12:
		# 	continue
		file_path = os.path.join(
			"base_classifiers", 
			"_".join(["Tox21_extended","label",f"{label_idx:02d}", "xgb"]),
			)

		valid_train_ind = np.where(~np.isnan(train_targets_scaled[:, label_idx]))[0]
		valid_test_ind = np.where(~np.isnan(test_targets_scaled[:, label_idx]))[0]
		X_train=train_features_scaled[valid_train_ind]
		X_test = test_features_scaled[valid_test_ind]
		y_train=train_targets_scaled[valid_train_ind, label_idx]
		y_test = test_targets_scaled[valid_test_ind, label_idx]
		train_sample_weight = compute_sample_weight("balanced", y_train)
		func=partial(tune_xgbc, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
		result = forest_minimize(func, space, random_state = 42, n_random_starts = 20, n_calls  = 25, verbose = 1)    

		with open(f"{file_path}_params_optimizer.ob", 'wb') as f:
			pickle.dump(result, f)	

		params = result['x']
		learning_rate, n_estimators, max_depth, min_child_weight, gamma, subsample, colsample_bytree = params

		clf = xgb.XGBClassifier(
			learning_rate = learning_rate,
			n_estimators = n_estimators, 
			max_depth = max_depth,
			min_child_weight = min_child_weight, 
			gamma = gamma,
			subsample = subsample,
			colsample_bytree = colsample_bytree, 
			objective = "reg:logistic", 
			n_jobs = -1
		)
		clf.fit(X_train, y_train, sample_weight = train_sample_weight)
		clf.save_model(f"{file_path}_model")

		train_predict = clf.predict_proba(X_train)[:,1]
		test_predict = clf.predict_proba(X_test)[:,1]
		train_results = model_eval(y_train, train_predict, target_scaler, "xgboost", label, "Train")
		test_results = model_eval(y_test, test_predict, target_scaler, "xgboost", label, "Test")
		file_updater(summary_csv_path, train_results, mode='a')
		file_updater(summary_csv_path, test_results, mode='a')
	sys.exit()
