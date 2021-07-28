#!/usr/bin/env python3
# File name: SynthData_10dim_experiments.py
# Compare dense and locality-sensitive models with increasing proportion of noise features


import os, sys
import numpy as np
import copy
import pandas as pd
import time

# code_folder = os.path.join("..", "..", "0_code")
# sys.path.append(code_folder)

from SynthData_experiment_helper import load_data, file_updater
from SynthData_experiment_helper import get_model_setup_params, get_model_compile_params, setup_callback_paths
from SynthData_experiment_helper import get_model, model_eval

n_points=1000
n_dim=10
n_clusters=25

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument(dest="model_type", help="model_type: dense, LS, LSwFW, LSwFW_ones, xgboost, RF")
args = parser.parse_args()

if __name__ == "__main__":
	model_type=args.model_type
	assert np.isin([model_type], ["dense", "LS", "LSwFW", "LSwFW_ones", "xgboost", "RF"])[0], "Choose model type as one of: 'dense', 'LS', 'LSwFW'."

	import tensorflow as tf
	gpus = tf.config.experimental.list_physical_devices('GPU')
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)

	tf.keras.backend.set_floatx('float32')	

	#Set up folder paths
	folder_name = "_".join([time.strftime("%y%m%d", time.localtime()), model_type])
	super_folder = os.path.join(f"SynthData_10dim_clusternoise_unbalanced", folder_name)
	try:
		os.mkdir(super_folder)
	except:
		pass
	summary_csv_path = os.path.join(super_folder, folder_name+".csv")
	header = ["Model", "n_noise_dim", "Metric", "Split", "Score"]
	file_updater(summary_csv_path, [header], mode='w+')

	n_noise_dim_list = range(1, 9)	
	for n_noise_dim in n_noise_dim_list:
		#Read datasets
		train_features, test_features, train_target, test_target, train_cluster_labels, test_cluster_labels, Fweights_train, Fweights_test = load_data(n_noise_dim)

		if model_type=="xgboost" or model_type=="RF":

			if model_type=="xgboost":
				from sklearn.utils.class_weight import compute_sample_weight
				import xgboost as xgb
				train_sample_weight = compute_sample_weight("balanced", train_target)				
				clf = xgb.XGBClassifier(
					n_jobs=-1
					)
				clf.fit(train_features, train_target, sample_weight = train_sample_weight)
				model_save_name = "_".join([folder_name, f"SynthData{str(n_noise_dim)}noisedim"])
				clf.save_model(os.path.join(super_folder, 
					model_save_name
					))				
			else:
				from sklearn.ensemble import RandomForestClassifier
				clf = RandomForestClassifier(class_weight="balanced")
				clf.fit(train_features, train_target)
			train_predict = clf.predict_proba(train_features)[:,1]
			test_predict = clf.predict_proba(test_features)[:,1]

		else:
			# Get model parameters
			learning_rate = 0.3
			model_setup_params = get_model_setup_params()
			model_setup_params['model_type']=model_type
			model_setup_params['n_feat'] = train_features.shape[1]    
			model_compile_params = get_model_compile_params(learning_rate)

			# Setup callback paths
			callbacks, checkpoint_path = setup_callback_paths(
				"val_auc",
				mode="max",
				model_name=f"{model_type}",
				dataset_name=f"SynthData{str(n_noise_dim)}noisedim",
				split_number="",
				super_folder=super_folder
			)

			if model_type=="LSwFW":
				# train_Fweights, test_Fweights = np.ones_like(train_features), np.ones_like(test_features)
				train_features = np.hstack([train_features, Fweights_train])
				test_features = np.hstack([test_features, Fweights_test])
			elif model_type=="LSwFW_ones":
				train_features = np.hstack([train_features, np.ones_like(Fweights_train)])
				test_features = np.hstack([test_features, np.ones_like(Fweights_test)])

			model=get_model(**model_setup_params)
			model.compile(**model_compile_params)
			model.fit(
				train_features,
				train_target, 
				validation_data=(test_features, test_target),
				epochs=100,
				callbacks=callbacks,
				verbose=2
			)

			# Evaluate model
			model.load_weights(checkpoint_path)
			train_predict = model(train_features).numpy()
			test_predict = model(test_features).numpy()

		results_train = model_eval(
			train_predict,
			train_target,
			target_scaler=None,
			model_name=model_type,
			n_noise_dim=n_noise_dim,
			split_name="Train"
		)
		results_test = model_eval(
			test_predict,
			test_target,
			target_scaler=None,
			model_name=model_type,
			n_noise_dim=n_noise_dim,
			split_name="Test"
		)
		file_updater(summary_csv_path, results_train, mode='a')
		file_updater(summary_csv_path, results_test, mode='a')
	sys.exit()