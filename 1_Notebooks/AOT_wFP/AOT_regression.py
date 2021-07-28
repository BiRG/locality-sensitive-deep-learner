import os
import sys
import pandas as pd
import numpy as np
import time
import csv

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.losses import BinaryCrossentropy

from AOTexperiment_helper import load_data, get_model, data_scaling, file_updater 
from AOTexperiment_helper import get_model_setup_params, get_model_compile_params, setup_callback_paths
from AOTexperiment_helper import cont_model_eval, get_cont_model_score

from sklearn.utils.class_weight import compute_sample_weight
def get_binned_sample_weight(y, n_bins, random_seed=123):
	n = len(y)
	counts, bins=np.histogram(y, bins=n_bins)
	ind = np.digitize(y, bins, right = True)
	sample_weight = compute_sample_weight("balanced", ind)
	return sample_weight


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(dest="model_type",
					help="model_type: dense, LS, LSwFW, LSwFW_ones, xgboost")
args = parser.parse_args()

if __name__ == "__main__":
	model_type = args.model_type
	assert np.isin([model_type], ["dense", "LS", "LSwFW", "LSwFW_ones", "xgboost"])[
		0], "Choose model type as one of: 'dense', 'LS', 'LSwFW', 'LSwFW_ones', 'xgboost'."

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

	#Convert LD50 to log10(LD50). Limit max LD50 to 10000 (in test dataset)
	train_targets = train_targets[:, 2].astype(np.float)
	ind = np.where(train_targets>10000.)[0]
	train_targets[ind]=10000.
	train_targets = np.log10(train_targets)
	test_targets = test_targets[:, 2].astype(np.float)
	ind = np.where(test_targets>10000.)[0]
	test_targets[ind]=10000.
	test_targets=np.log10(test_targets)
	valid_train_ind = np.where(~np.isnan(train_targets.astype(np.float)))[0]
	valid_test_ind = np.where(~np.isnan(test_targets.astype(np.float)))[0]

	# Scale data
	feature_scaler = StandardScaler()
	target_scaler = None
	# target_scaler = RobustScaler()
	cont_labels = [2]
	train_features_scaled, test_features_scaled, train_targets_scaled, test_targets_scaled, feature_scaler, target_scaler = data_scaling(
		feature_scaler,
		target_scaler,
		train_features,
		test_features,
		train_targets.astype(np.float32),
		test_targets.astype(np.float32)
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

	super_folder = "AOT_models"
	folder_name = "_".join([time.strftime("%y%m%d", time.localtime()),
							model_type,
							]
						   )
	summary_csv_path = os.path.join(super_folder,
									folder_name+"_regression_summary.csv"
									)
	header = ["Model", "Label", "Metric", "Type", "Score"]
	file_updater(summary_csv_path, [header], mode='w+')  

	# Get XGBoost parameters
	# if model_type=="xgboost":
		# with open("AOT_reg_xgb_params_optimizer.ob", 'rb') as f:
		# 	params=pickle.load(f)
		# params= params['x']

		# learning_rate = params[0] 
		# n_estimators = params[1] 
		# max_depth = params[2]
		# min_child_weight = params[3]
		# gamma = params[4]
		# subsample = params[5]
		# colsample_bytree = params[6]


	for idx, label_idx in enumerate(cont_labels):
		label=labels[label_idx]
		print(f"Starting on {label}")
		valid_train_ind = np.where(~np.isnan(train_targets_scaled))[0]
		valid_test_ind = np.where(~np.isnan(test_targets_scaled))[0]
		X_train = train_features_scaled[valid_train_ind]
		X_test = test_features_scaled[valid_test_ind]
		y_train = train_targets_scaled[valid_train_ind]
		y_test = test_targets_scaled[valid_test_ind]    	
		# train_sample_weight = compute_sample_weight("balanced", y_train)
		if model_type=="xgboost":
			try:
				os.mkdir(os.path.join(super_folder, "_".join([folder_name,"AOT","regression",label])))
			except:
				pass
			import xgboost as xgb
			model = xgb.XGBRegressor(
				# learning_rate = learning_rate, 
				n_estimators = 499, #n_estimators, 
				# max_depth=max_depth,
				# min_child_weight=min_child_weight,
				# gamma=gamma,
				# subsample=subsample,
				# colsample_bytree=colsample_bytree,
				objective="reg:squarederror",
				n_jobs=-1, 
				)
			model.fit(train_features_scaled[valid_train_ind], 
					train_targets_scaled[valid_train_ind],
					)
			model.save_model(os.path.join(super_folder, 
				"_".join([folder_name,"AOT","regression",label]), f"xgb_model_{label}"))

		else:
			#Get model parameters
			learning_rate=0.0001
			model_setup_params = get_model_setup_params()
			model_setup_params['n_feat']=n_feat
			model_setup_params['output_activation']="linear"
			model_compile_params = get_model_compile_params(learning_rate)
			model_compile_params['loss'] = tf.keras.losses.MeanSquaredError()
			model_compile_params['metrics'] = [tf.keras.metrics.RootMeanSquaredError()]

			#Setup callback paths
			callbacks, checkpoint_path = setup_callback_paths(
				"val_root_mean_squared_error", 
				mode="min", 
				model_name = f"{model_type}",
				dataset_name=f"AOT_regression",
				split_number=f"{label}",
				super_folder=super_folder
				)

			model = get_model(model_type, **model_setup_params)
			model.compile(**model_compile_params)
			model.fit(
				X_train, 
				y_train, 
				validation_data = (X_test, y_test),
				epochs=100, 
				callbacks=callbacks,
				verbose=2
				)

			#Evaluation
			model.load_weights(checkpoint_path)
		results_train, r2_train = cont_model_eval(model, 
			X_train, train_targets[valid_train_ind], target_scaler, 
			model_type, label, split_name="FirstFitTrain")
		results_test, r2_test = cont_model_eval(model, 
			X_test, test_targets[valid_test_ind], target_scaler, 
			model_type, label, split_name="FirstTest")
		model_score = get_cont_model_score(r2_train, r2_test, model_type, label)

		file_updater(summary_csv_path, results_train, mode='a')
		file_updater(summary_csv_path, results_test, mode='a')
		file_updater(summary_csv_path, model_score, mode='a')

		#Scramble last layer for second fit
		if model_type != "xgboost":
			weights=model.layers[-1].get_weights()
			weights[0] = model_setup_params['kernel_initializer'](weights[0].shape)
			weights[1] = model_setup_params['bias_initializer'](weights[1].shape)
			model.layers[-1].set_weights(weights)
			model.optimizer.learning_rate = 0.00001

			model.fit(X_train,
				y_train,
				validation_data = (X_test, y_test),
				epochs=100, 
				callbacks=callbacks,
				verbose=2
				)
			model.load_weights(checkpoint_path)
			results_train, r2_train = cont_model_eval(model, 
				X_train, 
				train_targets[valid_train_ind], 
				target_scaler, 
				model_type, label, split_name = "SecondFitTrain"
				)
			results_test, r2_test = cont_model_eval(model, 
				X_test, 
				test_targets[valid_test_ind], 
				target_scaler,
				model_type, label, split_name = "SecondFitTest"
				)
			model_score = get_cont_model_score(r2_train, r2_test, model_type, label)
			file_updater(summary_csv_path, results_train, mode='a')
			file_updater(summary_csv_path, results_test, mode='a')
			file_updater(summary_csv_path, model_score, mode='a')

	sys.exit()

	# 	r = []

	# 	train_predict = rgs.predict(train_features_scaled[valid_train_ind])
	# 	# train_predict = to_categorical(train_predict)
	# 	test_predict = rgs.predict(test_features_scaled[valid_test_ind], )
	# 	# test_predict = to_categorical(test_predict)

	# 	# categorical_y_train = to_categorical(y_train)

	# 	# diff = categorical_y_train.shape[1]-train_predict.shape[1]
	# 	# if diff>0:
	# 	#   train_predict = np.hstack([train_predict, np.zeros((train_predict.shape[0], diff))])
	# 	# diff = categorical_y_train.shape[1]-test_predict.shape[1]
	# 	# if diff>0:
	# 	#   test_predict=np.hstack([test_predict,np.zeros((test_predict.shape[0], diff))])

	# 	# cm_train = get_cm(categorical_y_train, train_predict)
	# 	cm_train = get_cm(y_train, train_predict)

	# 	sn_train, sp_train = get_sn_sp(cm_train)
	# 	print(f"Label {labels[label_idx]}: train NER={np.mean([sn_train, sp_train]):.3f}, train Sensitivity={sn_train:.3f}, train Specificity={sp_train:.3f}.")
	# 	r.append(["xgboost", labels[label_idx], "NER", "Train", np.mean([sn_train, sp_train])])
	# 	r.append(["xgboost", labels[label_idx], "Sensitivity", "Train", sn_train])
	# 	r.append(["xgboost", labels[label_idx], "Specificity", "Train", sp_train])

	# 	# cm_test = get_cm(to_categorical(y_test), test_predict)
	# 	cm_test = get_cm(y_test, test_predict)

	# 	sn_test, sp_test = get_sn_sp(cm_test)
	# 	print(f"Label {labels[label_idx]}: test NER={np.mean([sn_test, sp_test]):.3f}, test Sensitivity={sn_test:.3f}, test Specificity={sp_test:.3f}.")
	# 	r.append(["xgboost", labels[label_idx], "NER", "Test", np.mean([sn_test, sp_test])])
	# 	r.append(["xgboost", labels[label_idx], "Sensitivity", "Test", sn_test])
	# 	r.append(["xgboost", labels[label_idx], "Specificity", "Test", sp_test])

	# 	model_score = get_qual_model_score(sn_train, sp_train, sn_test, sp_test)
	# 	r.append(["xgboost", labels[label_idx], "model score", "TrainTest", model_score])
	# 	print(f"Label{labels[label_idx]}: model score={model_score:.3f}")

	# 	file_updater(summary_csv_path, r, mode='a')
	# sys.exit()