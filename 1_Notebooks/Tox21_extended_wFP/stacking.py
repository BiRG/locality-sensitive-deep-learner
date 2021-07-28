#!/usr/bin/env python3
# File name: stacking.py
# Author: XiuHuan Yap
# Contact: yapxiuhuan@gmail.com
"""Train stacking layer on Tox21_extended dataset, including tuning, from trained xgb models """

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

import time 

import dill as pickle

from Tox21_extended_experiment_helper import load_data, get_valid_ind, file_updater, data_scaling, tune_xgbc
from Tox21_extended_experiment_helper import model_eval

import argparse

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument("--train_predict_file", default = None, dest = 'train_predict_file', 
	help = "path to train predict file")
parser.add_argument("--test_predict_file", default = None, dest = 'test_predict_file',
	help = "path to test predict file")

args = parser.parse_args()

def gen_predict_init_files(labels, train_features, test_features, ):
	all_train_predict = []
	all_test_predict = []
	for label_idx, label in enumerate(labels):
		file_path = os.path.join(
			"base_classifiers", 
			"_".join(["Tox21_extended", "label", f"{label_idx:02d}", "xgb", "model"])
		)
		clf = xgb.XGBClassifier()
		clf.load_model(file_path)
		train_predict=clf.predict_proba(train_features)[:,1]
		test_predict = clf.predict_proba(test_features)[:,1]

		all_train_predict.append(train_predict)
		all_test_predict.append(test_predict)
	train_predict_init = np.vstack(all_train_predict).transpose()
	test_predict_init = np.vstack(all_test_predict).transpose()

	pd.DataFrame(train_predict_init, columns = labels).to_csv(os.path.join("base_classifiers", "base_classifiers_train_predict.csv"))
	pd.DataFrame(test_predict_init, columns = labels).to_csv(os.path.join("base_classifiers", "base_classifiers_test_predict.csv"))
	return train_predict_init, test_predict_init


if __name__ == "__main__":
	train_predict_file = args.train_predict_file
	test_predict_file = args.test_predict_file

	data_dict = load_data()
	train_features = data_dict['train_features']
	test_features = data_dict['test_features']
	train_targets = data_dict['train_targets'].values
	test_targets = data_dict['test_targets'].values
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
		train_targets.astype(np.float32),
		test_targets.astype(np.float32)
	)

	if train_predict_file is None:
		#Generate train predict file
		train_predict_init, test_predict_init = gen_predict_init_files(labels, 
			train_features_scaled, test_features_scaled)
	else:
		train_predict_init = pd.read_csv(train_predict_file, index_col=0).values
		test_predict_init = pd.read_csv(test_predict_file, index_col=0).values

	super_folder = "stacking_models"
	summary_csv_path = os.path.join(super_folder, 
		time.strftime("%y%m%d_summary.csv", time.localtime())
		)
	header = ["Model", "Label", "Split", "Metric", "Type", "Score"]
	file_updater(summary_csv_path, [header], mode="w+")

	#Inputs: Initial predictions for all labels, ground truth as target variables, and also for evaluation
	#Outputs: AUC score

	for label_idx, label in enumerate(labels):
		# if label_idx>3:
		# 	break
		file_path = os.path.join(
			"_".join(["Tox21_extended","label",f"{label_idx:02d}", "xgb"]),
		)

		valid_train_ind = get_valid_ind(train_targets[:, label_idx])
		valid_test_ind = get_valid_ind(test_targets[:, label_idx])
		X_train = train_predict_init[valid_train_ind]
		X_test = test_predict_init[valid_test_ind]
		y_train = train_targets_scaled[valid_train_ind, label_idx].astype(np.float32)
		y_test = test_targets_scaled[valid_test_ind, label_idx].astype(np.float32)

		train_sample_weight = compute_sample_weight("balanced", y_train)

		# # Creating a sample space in which the initial randomic search should be performed
		# space = [(1e-3, 1e-1, 'log-uniform'), # learning rate
		#           (100, 2000), # n_estimators
		#           (1, 10), # max_depth 
		#           (1, 6.), # min_child_weight 
		#           (0, 0.5), # gamma 
		#           (0.5, 1.), # subsample 
		#           (0.5, 1.)] # colsample_bytree 

		# from functools import partial
		# from skopt import forest_minimize
		# #Find optimal params
		# func = partial(
		# 	tune_xgbc, 
		# 	X_train = X_train, 
		# 	y_train = y_train, 
		# 	X_test = X_test, 
		# 	y_test = y_test
		# )
		# results = forest_minimize(func, space, random_state = 42, n_random_starts = 20, n_calls = 25, verbose = 1)
		# with open(os.path.join(super_folder, f"{file_path}_params_optimizer.ob"), 'wb') as f:
		# 	pickle.dump(results, f)
		# params = results['x']

		#Load xgb params
		# params_path = os.path.join("base_classifiers", 
		# 	file_path + "_params_optimizer.ob"
		# )
		# with open(params_path, 'rb') as f:
		# 	params = pickle.load(f)
		# params = params['x']

		# learning_rate, n_estimators, max_depth, min_child_weight, gamma, subsample, colsample_bytree = params

		# clf = xgb.XGBClassifier(
		# 	use_label_encoder = False,
		# 	objective = "binary:logistic",
		# 	eval_metric = "logloss",
		# 	learning_rate = learning_rate, 
		# 	n_estimators = n_estimators, 
		# 	max_depth = max_depth, 
		# 	min_child_weight = min_child_weight, 
		# 	gamma = gamma, 
		# 	subsample = subsample, 
		# 	colsample_bytree = colsample_bytree, 
		# 	n_jobs = -1, 
		# )
		# clf.fit(
		# 	X_train, 
		# 	y_train, 
		# 	sample_weight = train_sample_weight
		# )

		# clf.save_model(os.path.join(super_folder, file_path+"_model"))

		clf = xgb.XGBClassifier()
		clf.load_model(os.path.join(super_folder, file_path+"_model"))

		train_predict_final = clf.predict_proba(X_train)[:,1]
		test_predict_final = clf.predict_proba(X_test)[:,1]

		train_results = model_eval(y_train, train_predict_final, target_scaler, "xgboost_stacked", label, "Train")
		test_results = model_eval(y_test, test_predict_final, target_scaler, "xgboost_stacked", label, "Test")
		file_updater(summary_csv_path, train_results, mode='a')
		file_updater(summary_csv_path, test_results, mode='a')
	sys.exit()
