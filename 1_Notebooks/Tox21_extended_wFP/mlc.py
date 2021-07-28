#!/usr/bin/env python3
# File name: stacking.py
# Author: XiuHuan Yap
# Contact: yapxiuhuan@gmail.com
"""Train mlc given label partitioning on Tox21_extended dataset (including tuning) """

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

import time 

import dill as pickle

from Tox21_extended_experiment_helper import load_data, get_valid_ind, file_updater, data_scaling, tune_xgbc
from Tox21_extended_experiment_helper import to_multiclass, binarize, XGBTuneFit, XGBLoadFit
from Tox21_extended_experiment_helper import model_eval
from Tox21_extended_experiment_helper import remove_empty_classes, return_empty_classes

import argparse

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mlc_model", default = None, dest = 'mlc_model', 
	help = "MLC model type [ClassifierChains, LabelPowersets]")
parser.add_argument("-l", "--label_partitioning_file", default = None, dest = 'label_partitioning_file',
	help = "path to label partitioning file")

args = parser.parse_args()

from skmultilearn.ensemble import LabelSpacePartitioningClassifier
from skmultilearn.problem_transform import ClassifierChain, LabelPowerset

if __name__ =="__main__":
	mlc_model = args.mlc_model
	label_partitioning_file = args.label_partitioning_file
	label_partitioning_path = os.path.join("label_partitioning", label_partitioning_file)

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

	super_folder = "MLC_models"
	summary_csv_path = os.path.join(
		super_folder,
		time.strftime("%y%m%d", time.localtime())+f"{mlc_model}_{label_partitioning_file}_summary.csv"
	)
	header = ["Model", "Label", "Metric", "Split_name", "Score", "label_idx"]
	file_updater(summary_csv_path, [header], mode="w+")

	try:
		os.mkdir(os.path.join(super_folder, mlc_model))
	except:
		pass 

	with open(label_partitioning_path, 'rb') as f:
		lab_part = pickle.load(f)
	assert np.all(np.isin(list(range(len(labels))), np.concatenate(lab_part))), "All labels should be present in label partitioning"

	if mlc_model == "ClassifierChains":
		for label_set_idx, label_set in enumerate(lab_part):
			X_train_all = train_features_scaled
			X_test_all = test_features_scaled			
			for idx, label_idx in enumerate(label_set):
				label = labels[label_idx]
				file_path = os.path.join(
					super_folder, 
					mlc_model,
					"_".join(["Tox21_extended",
						label_partitioning_file,
						"labelset", f"{label_set_idx:02d}", 
						"label",f"{label_idx:02d}", 
						"xgb"]),
				)

				# params_path = os.path.join(
				# 	"base_classifiers",
				# 	f"Tox21_extended_label_{label_idx:02d}_xgb_params_optimizer.ob"
				# )

				valid_train_ind = get_valid_ind(train_targets_scaled[:, label_idx])
				valid_test_ind = get_valid_ind(test_targets_scaled[:, label_idx])
				X_train = X_train_all[valid_train_ind]
				X_test = X_test_all[valid_test_ind]
				y_train = train_targets_scaled[valid_train_ind, label_idx].astype(np.float32)
				y_test = test_targets_scaled[valid_test_ind, label_idx].astype(np.float32)	

				train_results, test_results, train_predict_all, test_predict_all = XGBTuneFit(X_train, X_test, y_train, y_test, target_scaler, 
						model_name = f"{mlc_model}_{label_partitioning_file}", 
						label_idx = label_idx, 
						label = label, 
						model_path = file_path, 
						X_train_all = X_train_all, 
						X_test_all = X_test_all,
					)
				# train_results, test_results, train_predict_all, test_predict_all = XGBLoadFit(X_train, X_test, y_train, y_test, target_scaler, params_path,  
				# 		model_name = f"{mlc_model}_{label_partitioning_file}", 
				# 		label_idx = label_idx, 
				# 		label = label, 
				# 		model_path = file_path, 
				# 		X_train_all=X_train_all, 
				# 		X_test_all = X_test_all,
				# 	)
				X_train_all = np.hstack([X_train_all, train_predict_all.reshape(-1,1)])
				X_test_all = np.hstack([X_test_all, test_predict_all.reshape(-1,1)])
				file_updater(summary_csv_path, train_results, mode = 'a')
				file_updater(summary_csv_path, test_results, mode='a')	

	elif mlc_model =="LabelPowersets":
		from functools import reduce
		for label_set_idx, label_set in enumerate(lab_part):
			if label_set_idx==0:
				continue
			l_ = "_".join([f"{l:02d}" for l in label_set])
			file_path = os.path.join(
					super_folder, 
					mlc_model,
					"_".join(["Tox21_extended",
						label_partitioning_file,
						"labelset", f"{label_set_idx:02d}", 
						"labels",l_, 
						"xgb"]),
				)
			valid_train_ind = reduce(np.intersect1d, (get_valid_ind(train_targets_scaled[:, label_idx]) for label_idx in label_set))
			valid_test_ind = reduce(np.intersect1d, (get_valid_ind(test_targets_scaled[:, label_idx]) for label_idx in label_set)) ##This is for selecting best parameters only, we will select valid indices for testing on a per-label basis
			X_train = train_features_scaled[valid_train_ind]
			X_test = test_features_scaled[valid_test_ind]
			y_train = train_targets_scaled[np.ix_(valid_train_ind, label_set)].astype(np.float32)
			y_test = test_targets_scaled[np.ix_(valid_test_ind, label_set)].astype(np.float32)
			y_train, n_classes = to_multiclass(y_train)
			y_test, _ = to_multiclass(y_test)
			y_train, ret_dict, y_test = remove_empty_classes(y_train, y_test)
			print(np.unique(y_train), ret_dict)
			# params_path = os.path.join(
			# 	"base_classifiers",
			# 	f"Tox21_extended_label_{label_idx:02d}_xgb_params_optimizer.ob"
			# )

			train_predict, test_predict = XGBTuneFit(
				X_train, X_test, y_train, y_test, target_scaler, 
				model_name = f"{mlc_model}_{label_partitioning_file}", 
				label_idx = label_set, 
				label = labels,
				model_path = file_path,
				multiclass = True,
				n_classes = n_classes, 
				X_train_all = train_features_scaled,
				X_test_all = test_features_scaled, 
				y_test_all = test_targets_scaled,
			)
			train_predict=return_empty_classes(train_predict, ret_dict)
			test_predict = return_empty_classes(test_predict, ret_dict)

			for idx, label_idx in enumerate(label_set):
				valid_train_ind=get_valid_ind(train_targets_scaled[:, label_idx])
				valid_test_ind = get_valid_ind(test_targets_scaled[:, label_idx])
				train_results = model_eval(
					train_targets_scaled[valid_train_ind, label_idx], 
					binarize(train_predict[valid_train_ind], len(label_set),idx), 
					target_scaler, 
					model_name = f"{mlc_model}_{label_partitioning_file}",
					label=labels[label_idx],
					split_name="Train",
					)
				train_results[0].append(label_idx)
				test_results = model_eval(
					test_targets_scaled[valid_test_ind, label_idx],
					binarize(test_predict[valid_test_ind], len(label_set), idx),
					target_scaler, 
					model_name = f"{mlc_model}_{label_partitioning_file}",
					label=labels[label_idx], 
					split_name="Test")
				test_results[0].append(label_idx)
				file_updater(summary_csv_path, train_results, mode='a')
				file_updater(summary_csv_path, test_results, mode='a')

			# train_results, test_results = XGBLoadFit(
			# 	X_train, X_test, y_train, y_test, target_scaler, params_path,
			# 	model_name = f"{mlc_model}_{label_partitioning_file}", 
			# 	label_idx = label_ind, 
			# 	label = labels,
			# 	model_path = file_path,
			# 	multiclass = True,
			# 	n_classes = n_classes, 
			# 	X_test_all = test_features_scaled, 
			# 	y_test_all = test_targets_scaled,
			# )			
			# file_updater(summary_csv_path, train_results, mode='a')
			# file_updater(summary_csv_path, test_results, mode='a')

	sys.exit()
