#!/usr/bin/env python3
# File name: Stacking.py
# Author: XiuHuan Yap
# Contact: yapxiuhuan@gmail.com
"""Tuning Stacking layer on Acute oral toxicity dataset, from train/test predictions of initial model"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

import time

import dill as pickle

from sklearn.metrics import mean_squared_error, roc_auc_score

def get_valid_ind(y):
	ind = np.where(~np.isnan(y.astype(np.float32)))[0]
	return ind

def tune_xgbc(params, X_train, y_train, X_test, y_test, regression=False, 
	objective = "reg:logistic", eval_metric='logloss'):
# Implementation learned on a lesson of Mario Filho (Kagle Grandmaster) for parametes optmization.
# Link to the video: https://www.youtube.com/watch?v=WhnkeasZNHI

	"""Function to be passed as scikit-optimize minimizer/maximizer input

	Parameters:
	Tuples with information about the range that the optimizer should use for that parameter, 
	as well as the behaviour that it should follow in that range.

	Returns:
	float: the metric that should be minimized. If the objective is maximization, then the negative 
	of the desired metric must be returned. In this case, the negative AUC average generated by CV is returned.
	"""


	#Hyperparameters to be optimized
	print(params)
	learning_rate = params[0] 
	n_estimators = params[1] 
	max_depth = params[2]
	min_child_weight = params[3]
	gamma = params[4]
	subsample = params[5]
	colsample_bytree = params[6]


	#Model to be optimized
	if regression:
		sample_weight = None  	
	
		mdl = xgb.XGBRegressor(learning_rate = learning_rate, 
							n_estimators = n_estimators, 
							max_depth = max_depth, 
							min_child_weight = min_child_weight, 
							gamma = gamma, 
							subsample = subsample, 
							colsample_bytree = colsample_bytree, 
							objective=objective,
							use_label_encoder=False,
							seed = 42
							)
	else:
		sample_weight = compute_sample_weight("balanced", y_train)    	
		mdl = xgb.XGBClassifier(learning_rate = learning_rate, 
							n_estimators = n_estimators, 
							max_depth = max_depth, 
							min_child_weight = min_child_weight, 
							gamma = gamma, 
							subsample = subsample, 
							colsample_bytree = colsample_bytree, 
							objective=objective,
							use_label_encoder=False,
							eval_metric=eval_metric,
							seed = 42)



	# #Cross-Validation in order to avoid overfitting
	# auc = cross_val_score(mdl, X_train_selected, y_train, cv = 10, scoring = 'roc_auc')

	mdl.fit(X_train, y_train, sample_weight = sample_weight)
	if regression:
		test_predict = mdl.predict(X_test)
		res = mean_squared_error(y_test, test_predict, squared=False)
	else:
		test_predict = mdl.predict_proba(X_test)
		y_test=y_test.astype(int)
		if test_predict.shape[1]==2:
			test_predict=test_predict[:, 1]
		res = roc_auc_score(y_test, test_predict, multi_class = 'ovr')

	print(res.mean())
	# as the function is minimization (forest_minimize), we need to use the negative of the desired metric (AUC)
	return -res.mean()

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

	from CoMPARA_experiment_helper import load_data

	data_dict = load_data()
	# train_features = data_dict['train_features']
	# test_features = data_dict['test_features']
	train_targets = data_dict['train_targets']
	test_targets = data_dict['test_targets']
	# train_Fweights = data_dict['train_Fweights']
	# test_Fweights = data_dict['test_Fweights']
	labels = data_dict['labels']

	binary_labels=[0,2,4]

	super_folder = "CoMPARA_Stacking_models"
	folder_name = "_".join([time.strftime("%y%m%d", time.localtime()),
							model_type, "params"
							])
	file_path = os.path.join(super_folder, folder_name)

	#Code adapted from Kaggle discussion board: https://www.kaggle.com/general/17120
	#Use random forest to optimize
	from skopt import forest_minimize
	from sklearn.utils.class_weight import compute_sample_weight
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
	for idx, label_idx in enumerate(binary_labels):
		label=labels[label_idx]
		valid_train_ind = get_valid_ind(train_targets.values[:, label_idx])
		valid_test_ind = get_valid_ind(test_targets.values[:, label_idx])
		y_train = train_targets.values[valid_train_ind, label_idx].astype(int)
		y_test = test_targets.values[valid_test_ind, label_idx].astype(int)
		
		regression = False
		objective = "binary:logistic"
		eval_metric = "logloss"

		func=partial(tune_xgbc, 
			X_train=train_predict_init[valid_train_ind], 
			y_train=y_train, 
			X_test=test_predict_init[valid_test_ind], 
			y_test=y_test, 
			objective=objective,
			regression = regression,
			eval_metric=eval_metric
			)

		result = forest_minimize(func, space, random_state = 42, n_random_starts = 20, 
			n_calls  = 25, verbose = 1)    

		with open(file_path+f"_{label}.ob", 'wb') as f:
			pickle.dump(result, f)