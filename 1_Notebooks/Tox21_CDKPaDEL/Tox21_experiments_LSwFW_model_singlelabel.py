#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import time
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.initializers import VarianceScaling, HeNormal
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.losses import BinaryCrossentropy

from Tox21experiment_helper import *

from sklearn.utils.class_weight import compute_sample_weight

import sys
sys.path.append(os.path.join("..","..","0_code"))

if __name__ == "__main__":
	import csv

	labels, all_train_features, test_features, all_train_targets, test_targets, train_id_df, test_id_df, all_train_Fweights, test_Fweights = load_data(
		load_all_features=True, 
		load_v1_dataset = False
		)

	import tensorflow as tf
	gpus = tf.config.experimental.list_physical_devices('GPU')
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)
	tf.keras.backend.set_floatx('float32')

	from algorithms import attention_model
	from sklearn.preprocessing import StandardScaler, MinMaxScaler

	from tf_helpers import get_weights_dicts, BinaryCrossEntropyIgnoreNan
	from tf_helpers import SimilarityBatchingDataset
	from tf_helpers import AveragedAUCIgnoreNan

	learning_rate = 0.05 #Fix this later
	lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

# ADD ARGPARSER for setup kwargs??
	model_setup_params = get_model_setup_params()
	weights_dicts = get_weights_dicts(all_train_targets)
	model_compile_params = get_model_compile_params(weights_dicts, learning_rate)
	dataset_params = get_dataset_params()
	n_batch=dataset_params['n_batch']
	n_buffer=dataset_params['n_buffer']

	params = {
	'n_attention_out':3,
	'n_attention': 10,
	'n_out': 1,
	}

	import time
	import dill as pickle

	for idx in [0]:
		#Update Model setup parameters
		for key in params.keys():
			model_setup_params[key] = params[key]
		# split_name = "__".join([f"{key}_{params[key]}" for key in params.keys()])
		n_feat = all_train_features.shape[1]
		model_setup_params['n_feat'] = n_feat 	 
		model_setup_params['kernel_initializer']=HeNormal()
		# model_compile_params['metrics'] = AveragedAUCIgnoreNan(
		# 		num_labels=1, 
		# 	)
		model_compile_params['loss'] = BinaryCrossentropy(label_smoothing=0.1)	
		model_compile_params['metrics'] = "AUC"

		#Data scaling
		feature_scaler = StandardScaler()
		target_scaler = None
		
		_ = data_scaling(feature_scaler, 
					 target_scaler, 
					 all_train_features, 
					 test_features, 
					 all_train_targets,
					 test_targets)
		all_train_features_scaled, test_features_scaled, all_train_targets_scaled, test_targets_scaled, feature_scaler, target_scaler=_
		# train_features_scaled = all_train_features_scaled[train_ind, :]
		# val_features_scaled = all_train_features_scaled[val_ind, :]
		# train_targets_scaled = all_train_targets_scaled[train_ind]
		# val_targets_scaled = all_train_targets_scaled[val_ind]
		
		# Load Feature weighting
		# train_Fweights = all_train_Fweights[train_ind,:]
		# val_Fweights = all_train_Fweights[val_ind,:]
		
		# Tensor casting
		# train_targets_scaled = tf.cast(train_targets_scaled, tf.float32)
		# val_targets_scaled = tf.cast(val_targets_scaled, tf.float32)
		# test_targets_scaled = tf.cast(test_targets_scaled, tf.float32)
		
		train_features_scaled = all_train_features_scaled
		val_features_scaled = None

		train_Fweights = all_train_Fweights
		val_Fweights = None

		train_targets_scaled = tf.cast(all_train_targets_scaled, tf.float32)
		val_targets_scaled = None
		test_targets_scaled = tf.cast(test_targets_scaled, tf.float32)		

		for label_idx, label in enumerate(labels):
			# Setup tf datasets
			# Not using tf datasets any more as we introduce sample_weight into tf.keras.model.fit() method (instead of through custom loss)
			valid_train = np.where(~np.isnan(train_targets_scaled[:, label_idx]))[0]
			valid_test = np.where(~np.isnan(test_targets_scaled[:, label_idx]))[0]

			# ds_dict = setup_tf_datasets(
			# 	train_features_scaled[valid_train], 
			# 	train_targets_scaled.numpy()[valid_train,label_idx],
			# 	None, #val_features_scaled,
			# 	None, #val_targets_scaled[:, label_idx], 
			# 	test_features_scaled[valid_test],
			# 	test_targets_scaled.numpy()[valid_test, label_idx], 
			# 	all_train_Fweights[valid_train], #train_Fweights,
			# 	None, #val_Fweights,
			# 	test_Fweights[valid_test], #test_Fweights, 
			# 	shuffle_buffer = n_buffer,
			# 	batch_size = n_batch,
			# 	)
			#Update metric
			# model_compile_params['loss']= BinaryCrossEntropyIgnoreNan(
			# 	[weights_dicts[label_idx]])

			sample_weight = compute_sample_weight("balanced", 
				train_targets_scaled.numpy()[valid_train, label_idx])

			# def load_ds_dict(ds_dict):
			# 	train_ds = ds_dict.get('train_ds')
			# 	val_ds = ds_dict.get('val_ds')
			# 	test_ds = ds_dict.get('test_ds')
			# 	return train_ds, val_ds, test_ds

			#Get callbacks and save paths
			super_folder = os.path.join("SingleLabelModels", 
				time.strftime("%y%m%d_Tox21", time.localtime())
				)
			callbacks, checkpoint_path = setup_callback_paths("val_auc",
			#"val_averaged_auc_ignore_nan",
															  mode="max",
															  model_name="LSwFW",
															  dataset_name=f"Tox21_{label}",
															  split_number="",
															  super_folder=super_folder,
															  )
			
			callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_scheduler))
			
			#Save run info
			to_save = {#'train_ind': train_ind,
					   # 'val_ind': val_ind,
					   'feature_scaler': feature_scaler,
					   'target_scaler': target_scaler,
					   }
			with open(os.path.join(checkpoint_path, "..",
								   "input_info.ob"), 'wb') as f:
				pickle.dump(to_save, f)

			to_save = {
				"_call": get_attention_model,
				"model_setup_params": model_setup_params,
				"model_compile_params": model_compile_params,
				"dataset_params": dataset_params
			}
			with open(os.path.join(checkpoint_path, "..",
								   "model_params.ob"), 'wb') as f:
				pickle.dump(to_save, f)

			output_csv = os.path.join(checkpoint_path, "output.csv")


			header = ["Model", "Split", "Label", "Metric", "Type", "Score"]
			with open(output_csv, 'w+', newline='', encoding='utf-8') as f:
				writer = csv.writer(f, delimiter = ',')
				writer.writerow(header)
			if label_idx==0:
				summary_csv = os.path.join(checkpoint_path, "..", "..",  
					time.strftime("%y%m%d_LSwFW_summary_lr0_005.csv", time.localtime()))				
				with open(summary_csv, 'w+', newline='', encoding='utf-8') as f:
					writer = csv.writer(f, delimiter=',')
					writer.writerow(header)

			# Set up model
			LS_model = get_attentionwFW_model(**model_setup_params
												 )

			LS_model.compile(**model_compile_params)

			# Fit and train model
			# train_ds, val_ds, test_ds = load_ds_dict(ds_dict)
			train_tensor_scaled = np.hstack([train_features_scaled[valid_train], all_train_Fweights[valid_train]])
			test_tensor_scaled = np.hstack([test_features_scaled[valid_test], 
									test_Fweights[valid_test]])
			y_train=train_targets_scaled.numpy()[valid_train, label_idx]
			y_test = test_targets_scaled.numpy()[valid_test, label_idx]

			LS_model.fit(#train_ds,
				train_tensor_scaled,
				y_train,
				sample_weight=sample_weight,
								validation_data = (test_tensor_scaled, 
								),
								epochs=200,
								callbacks=callbacks,
								verbose=2
								)

			# Evaluate model
			LS_model.load_weights(checkpoint_path)
			# LS_model.save(os.path.join(checkpoint_path, "..", "saved_model"))

			test_predict = LS_model.predict(test_tensor_scaled)
			if target_scaler is not None:
				test_predict = target_scaler.inverse_transform(test_predict)

			from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
			test_predict_class = tf.cast(tf.math.greater(test_predict[ind, ], 0.5),
				dtype = tf.int32)
			auc = roc_auc_score(y_test, test_predict)
			f1 = f1_score(y_test, test_predict_class)
			acc = accuracy_score(y_test, test_predict_class)

			print(f"{label}: AUC = {auc:.3f}; f1 = {f1:.3f}; acc = {acc:.3f}.")

			with open(output_csv, 'a', newline='', encoding='utf-8') as f:
				writer = csv.writer(f, delimiter = ',')
				writer.writerow(["LSwFW_model", idx, label, "AUC", "FirstFit_Test", auc])
				writer.writerow(["LSwFW_model", idx, label, "f1", "FirstFit_Test", f1])
				writer.writerow(["LSwFW_model", idx, label, "acc", "FirstFit_Test", acc])

			with open(summary_csv, 'a', newline='', encoding='utf-8') as f:
				writer = csv.writer(f, delimiter=',')
				writer.writerow(["LSwFW_model", idx, label, "AUC", "FirstFit_Test", auc])
			
			#Second fit with lower learning rate
			LS_model.optimizer.learning_rate = 0.005
			LS_model.fit(train_tensor_scaled, 
				y_train, 
				sample_weight = sample_weight, 
				validation_data = (test_tensor_scaled, y_test)
				)
			LS_model.load_weights(checkpoint_path)
			test_predict_class = tf.cast(tf.math.greater(test_predict[ind, ], 0.5),
				dtype = tf.int32)
			auc = roc_auc_score(y_test, test_predict)
			f1 = f1_score(y_test, test_predict_class)
			acc = accuracy_score(y_test, test_predict_class)

			print(f"{label}: AUC = {auc:.3f}; f1 = {f1:.3f}; acc = {acc:.3f}.")

			with open(output_csv, 'a', newline='', encoding='utf-8') as f:
				writer = csv.writer(f, delimiter = ',')
				writer.writerow(["LSwFW_model", idx, label, "AUC", "SecondFit_Test", auc])
				writer.writerow(["LSwFW_model", idx, label, "f1", "SecondFit_Test", f1])
				writer.writerow(["LSwFW_model", idx, label, "acc", "SecondFit_Test", acc])

			with open(summary_csv, 'a', newline='', encoding='utf-8') as f:
				writer = csv.writer(f, delimiter=',')
				writer.writerow(["LSwFW_model", idx, label, "AUC", "SecondFit_Test", auc])

			del LS_model
	sys.exit()