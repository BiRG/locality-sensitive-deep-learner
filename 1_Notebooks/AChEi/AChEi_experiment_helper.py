#!/usr/bin/env python3
# File name: AChEi_experiment_helper.py

import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

import os, sys
sys.path.append(os.path.join("..", "..", "0_code"))

import tensorflow as tf
import numpy as np
from algorithms import attention_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.regularizers import l1, l2
from functools import partial 
import dill as pickle
import csv


def load_data():
	with open("AChEi_extendedDataset_allFeatures.ob", 'rb') as f:
		r = pickle.load(f)
	return r

def file_updater(file_path, rows, mode='a'):
	with open(file_path, mode, newline='', encoding='utf-8') as f:
		writer=csv.writer(f)
		for row in rows:
			writer.writerow(row)

def data_scaling(feature_scaler,
                 target_scaler,
                 train_features,
                 test_features,
                 train_target,
                 test_target
                 ):
    # Scaling
    feature_scaler.fit(train_features)
    train_features_scaled = feature_scaler.transform(train_features)
    test_features_scaled = feature_scaler.transform(test_features)
    if target_scaler is not None:
        target_scaler.fit(np.expand_dims(train_target, axis=1))
        train_target_scaled = target_scaler.transform(
            np.expand_dims(train_target, axis=1)).flatten()
        test_target_scaled = target_scaler.transform(
            np.expand_dims(test_target, axis=1)).flatten()
    else:
        train_target_scaled = train_target
        test_target_scaled = test_target
    return train_features_scaled, test_features_scaled, train_target_scaled, test_target_scaled, feature_scaler, target_scaler

def setup_callback_paths(monitor,
						 mode,
						 model_name,
						 dataset_name,
						 split_number=0,
						 super_folder=None,
						 ):
	if isinstance(split_number, (int, float)):
		split_name="split{:02d}".format(split_number)
	else:
		split_name=split_number
	save_folder = "_".join([time.strftime("%y%m%d", time.localtime()),
							model_name,
							dataset_name,
							split_name
							])
	if super_folder is not None:
		save_folder = os.path.join(super_folder,
								   save_folder
								   )
		try:
			os.mkdir(super_folder)
		except OSError as error:
			print(error)
	checkpoint_path = os.path.join(save_folder,
								   "model_checkpoint")
	csv_filename = os.path.join(checkpoint_path, "training_log.csv")
	try:
		os.mkdir(save_folder)
	except OSError as error:
		print(error)
	try:
		os.mkdir(checkpoint_path)
	except OSError as error:
		print(error)

	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
													 monitor=monitor,
													 mode=mode,
													 save_best_only=True,
													 save_weights_only=True,
													 verbose=1
													 )
	csvlogger_callback = tf.keras.callbacks.CSVLogger(filename=csv_filename,
													  append=True
													  )
	return [cp_callback, csvlogger_callback], checkpoint_path

def get_model_setup_params():
	model_setup_params = {
		"n_attention": 10,
		"n_attention_hidden": 512,
		'n_feat': None,
		"n_out": 1,
		"n_concat_hidden": 512,
		"concat_activation": tf.keras.layers.LeakyReLU(alpha=0.1),
		"n_attention_out": 3,
	    "kernel_initializer": tf.keras.initializers.Orthogonal(),
		# "kernel_initializer": tf.keras.initializers.HeNormal(),
	#     "bias_initializer": tf.keras.initializers.Zeros(),
		"bias_initializer": tf.keras.initializers.Constant(value=0.1),     # So that we have weights to train on each LeakyReLU neuron
	    "attention_kernel_initializer": tf.keras.initializers.Orthogonal(),
		# "attention_kernel_initializer": tf.keras.initializers.HeNormal(),
	#     "attention_bias_initializer": tf.keras.initializers.Zeros(),
		"attention_bias_initializer": tf.keras.initializers.Constant(value=0.1),
		"attention_hidden_activation": tf.keras.layers.LeakyReLU(alpha=0.1),
	#     "attention_hidden_activation": "selu",
		"attention_output_activation": "sigmoid", 
		"n_hidden": 512,
	#     "hidden_activation": "selu", 
		"hidden_activation": tf.keras.layers.LeakyReLU(alpha=0.1),
		"kernel_regularizer": l2(1E-5),
		"bias_regularizer": l2(1E-5),
		"random_seed": 123
	}
	return model_setup_params


def get_model_compile_params(learning_rate):
	model_compile_params = {
		"optimizer": tf.keras.optimizers.Adadelta(learning_rate=learning_rate,
												  clipvalue=0.5,
												  clipnorm=1.0
												  ),
		"loss": tf.keras.losses.MAE,
		"metrics": ["mae", "mse",
					tf.keras.metrics.RootMeanSquaredError()
					]
	}
	return model_compile_params


def get_dataset_params():
	dataset_params = {
		"n_batch": 8,
		"n_buffer": 100,
	}
	return dataset_params


def _get_model(model_type,
			   n_attention,
			   n_attention_hidden,
			   n_feat,
			   n_out,
			   n_concat_hidden,
			   concat_activation,
			   n_attention_out,
			   kernel_initializer,
			   bias_initializer,
			   attention_kernel_initializer,
			   attention_bias_initializer,
			   attention_hidden_activation,
			   attention_output_activation,
			   n_hidden,
			   hidden_activation,
			   kernel_regularizer,
			   bias_regularizer,
			   random_seed=123):
	tf.random.set_seed(random_seed)
	if model_type == "LSwFW" or model_type=="LSwFW_ones":
		input_shape = (n_feat*2,)
	else:
		input_shape = (n_feat,)
	input_layer = Input(shape=input_shape)

	if model_type == "dense":
		print(f"First dense layer with {n_attention*n_attention_hidden*n_attention_out} hidden units")
		first_layer = Dense(n_attention*n_attention_hidden*n_attention_out,
							activation=hidden_activation,
							kernel_initializer=kernel_initializer,
							kernel_regularizer=kernel_regularizer,
							bias_initializer=bias_initializer,
							bias_regularizer=bias_regularizer,
							)(input_layer)

		layer0 = Dense(n_concat_hidden,
					   activation=hidden_activation,
					   kernel_initializer=kernel_initializer,
					   kernel_regularizer=kernel_regularizer,
					   bias_initializer=bias_initializer,
					   bias_regularizer=bias_regularizer,
					   )(first_layer)
	elif model_type == "LS":
		layer0 = attention_model.ConcatAttentions(
			n_attention=n_attention,
			n_attention_hidden=n_attention_hidden,
			n_attention_out=n_attention_out,
			n_feat=n_feat,
			n_hidden=n_concat_hidden,
			activation=concat_activation,
			kernel_initializer=kernel_initializer,
			bias_initializer=bias_initializer,
			kernel_regularizer=l2(1E-5),
			bias_regularizer=l2(1E-5),
			attention_kernel_initializer=attention_kernel_initializer,
			attention_bias_initializer=attention_bias_initializer,
			attention_hidden_activation=attention_hidden_activation,
			attention_output_activation=attention_output_activation,
			batch_norm_kwargs={"trainable": False, "renorm": True},
		)(input_layer)
	elif model_type == "LSwFW" or model_type=="LSwFW_ones":
		layer0 = attention_model.ConcatAttentionswFeatWeights(
			n_attention=n_attention,
			n_attention_hidden=n_attention_hidden,
			n_attention_out=n_attention_out,
			n_feat=n_feat,
			n_hidden=n_concat_hidden,
			activation=concat_activation,
			kernel_initializer=kernel_initializer,
			bias_initializer=bias_initializer,
			kernel_regularizer=l2(1E-5),
			bias_regularizer=l2(1E-5),
			attention_kernel_initializer=kernel_initializer,
			attention_bias_initializer=bias_initializer,
			attention_hidden_activation=attention_hidden_activation,
			attention_output_activation=attention_output_activation,
			batch_norm_kwargs={"trainable": False, "renorm": True},
		)(input_layer)
	dense_layer1 = Dense(n_hidden,
						 activation=hidden_activation,
						 kernel_initializer=kernel_initializer,
						 kernel_regularizer=kernel_regularizer,
						 bias_initializer=bias_initializer,
						 bias_regularizer=bias_regularizer,
						 )(layer0)
	dropout1 = Dropout(0.1)(dense_layer1)
	# batchnorm1 = BatchNormalization(trainable=False,
	# 								renorm=True
	# 								)(dropout1)
	dense_layer2 = Dense(n_hidden,
						 activation=hidden_activation,
						 kernel_initializer=kernel_initializer,
						 kernel_regularizer=kernel_regularizer,
						 bias_initializer=bias_initializer,
						 bias_regularizer=bias_regularizer
						 )(dropout1)
	dropout2 = Dropout(0.1)(dense_layer2)
	# batchnorm2 = BatchNormalization(trainable=False,
	# 								renorm=True
	# 								)(dropout2)
	dense_layer3 = Dense(n_hidden,
						 activation=hidden_activation,
						 kernel_initializer=kernel_initializer,
						 kernel_regularizer=kernel_regularizer,
						 bias_initializer=bias_initializer,
						 bias_regularizer=bias_regularizer,
						 )(dropout2)
	output_layer = Dense(n_out, activation="linear")(dense_layer3)

	tf_model = tf.keras.Model(inputs=input_layer,
							  outputs=output_layer
							  )
	return tf_model


def get_dense_model(**kwargs):
	model_type = "dense"
	return _get_model(model_type,
					  **kwargs
					  )


def get_attention_model(**kwargs):
	model_type = "LS"
	return _get_model(model_type,
					  **kwargs)


def get_attentionwFW_model(**kwargs):
	model_type = "LSwFW"
	return _get_model(model_type,
					  **kwargs
					  )

def model_eval(tf_model, 
	X, y, target_scaler, model_name, fold_idx, split_name="Train", 
):
	if model_name=="xgboost":
		predict = target_scaler.inverse_transform(
			tf_model.predict(X).reshape(-1,1)).flatten()
	else:
		predict = target_scaler.inverse_transform(
			tf_model(X).numpy()
		).flatten()


	rmse=mean_squared_error(y, predict, squared=False)
	r2=r2_score(y, predict)

	results=[
		[model_name, fold_idx, 'RMSE', split_name, rmse], 
		[model_name, fold_idx, 'Rsquared', split_name, r2]
	]
	return results