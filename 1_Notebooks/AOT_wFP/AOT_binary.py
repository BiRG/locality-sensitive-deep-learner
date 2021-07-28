#!/usr/bin/env python3
# File name: AOT_binary.py
# Author: XiuHuan Yap
# Contact: yapxiuhuan@gmail.com
"""Train/Test Locality-sensitive learner (with and without feature weights) and deep learner on Acute oral toxicity dataset"""

import os
import sys
import pandas as pd
import numpy as np
import time
import csv

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.losses import BinaryCrossentropy

from AOTexperiment_helper import load_data, get_model, data_scaling, file_updater 
from AOTexperiment_helper import get_model_setup_params, get_model_compile_params, setup_callback_paths
from AOTexperiment_helper import model_eval, model_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(dest="model_type",
					help="model_type: dense, LS, LSwFW, LSwFW_ones")
args = parser.parse_args()

if __name__ == "__main__":
	model_type = args.model_type
	assert np.isin([model_type], ["dense", "LS", "LSwFW", "LSwFW_ones"])[
		0], "Choose model type as one of: 'dense', 'LS', 'LSwFW'."

	import tensorflow as tf
	gpus = tf.config.experimental.list_physical_devices('GPU')
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)

	tf.keras.backend.set_floatx('float32')

	data_dict = load_data("data")
	train_features = data_dict['train_features']
	test_features = data_dict['test_features']
	train_targets = data_dict['train_targets']
	test_targets = data_dict['test_targets']
	train_Fweights = data_dict['train_Fweights']
	test_Fweights = data_dict['test_Fweights']
	labels = data_dict['labels']

	# Scale data
	feature_scaler = StandardScaler()
	target_scaler = None
	binary_labels = [0, 1]
	train_features_scaled, test_features_scaled, train_targets_scaled, test_targets_scaled, feature_scaler, target_scaler = data_scaling(
		feature_scaler,
		target_scaler,
		train_features,
		test_features,
		train_targets[:, binary_labels].astype(np.float32),
		test_targets[:, binary_labels].astype(np.float32)
	)
	n_feat = train_features_scaled.shape[1]

	if model_type == "LSwFW":
		train_features_scaled = np.hstack(
			[train_features_scaled, train_Fweights])
		test_features_scaled = np.hstack(
			[test_features_scaled, test_Fweights])

	elif model_type == "LSwFW_ones":
		train_features_scaled = np.hstack(
			[train_features_scaled, np.ones_like(train_features_scaled)])
		test_features_scaled = np.hstack(
			[test_features_scaled, np.ones_like(test_features_scaled)])

	model_setup_params = get_model_setup_params()
	model_setup_params['n_out'] = 1
	model_setup_params['n_feat'] = n_feat
	model_compile_params = get_model_compile_params(learning_rate=0.0001)

	super_folder = "AOT_models"
	folder_name = "_".join([time.strftime("%y%m%d", time.localtime()),
							model_type,
							"binary",
							"summary"]
						   )
	summary_csv_path = os.path.join(super_folder,
									folder_name+".csv"
									)
	header = ["Model", "Label", "Metric", "Type", "Score"]
	file_updater(summary_csv_path, [header], mode='w+')

	for idx, label_idx in enumerate(binary_labels):
		label = labels[label_idx]
		callbacks, checkpoint_path = setup_callback_paths('val_auc',
														  mode='max',
														  model_name=model_type,
														  dataset_name=f"AOT_wFP_{label}",
														  split_number="",
														  super_folder=super_folder
														  )

		model = get_model(model_type=model_type, **model_setup_params)
		model.compile(**model_compile_params)

		valid_train_ind = np.where(~np.isnan(train_targets_scaled[:, idx]))[0]
		valid_test_ind = np.where(~np.isnan(test_targets_scaled[:, idx]))[0]
		X_train = train_features_scaled[valid_train_ind]
		X_test = test_features_scaled[valid_test_ind]
		y_train = train_targets_scaled[valid_train_ind, idx]
		y_test = test_targets_scaled[valid_test_ind, idx]

		from sklearn.utils.class_weight import compute_sample_weight
		sample_weight = compute_sample_weight("balanced", y_train)

		model.fit(X_train, y_train,
			  validation_data=(X_test, y_test),
			  epochs=100,
			  sample_weight=sample_weight,
			  callbacks=callbacks,
			  verbose=2
			  )

		model.load_weights(checkpoint_path)
		# model.save(os.path.join(checkpoint_path, "..", "saved_model"))

		# Model Evaluation
		r_train, sn_train, sp_train = model_eval(
			model, X_train, y_train, target_scaler, model_type, label, split_name="Train")
		r_test, sn_test, sp_test = model_eval(
			model, X_test, y_test, target_scaler, model_type, label, split_name="Test")
		r_modelscore = model_score(
			sn_train, sp_train, sn_test, sp_test, model_type, label, split_name="TrainTest")
		file_updater(summary_csv_path, r_train, mode='a')
		file_updater(summary_csv_path, r_test, mode='a')
		file_updater(summary_csv_path, r_modelscore, mode='a')
	sys.exit()
