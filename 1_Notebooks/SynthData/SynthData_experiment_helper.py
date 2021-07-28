#!/usr/bin/env python3
# File name: SynthData_experiment_helper.py

import os
import sys
import pandas as pd
import numpy as np
import time

from sklearn.metrics import roc_auc_score, accuracy_score

code_folder = os.path.join("..", "..", "0_code")
sys.path.append(code_folder)

import tensorflow as tf
import numpy as np
from algorithms import attention_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.regularizers import l1, l2
from functools import partial 
import dill as pickle
import csv

def load_data(n_noise_dim):
	SynthDataFolder = "SynthData_10dim_clusternoise_unbalanced"
	file_path = os.path.join(SynthDataFolder, 
		f"SynthData_10dim_clusternoise_{str(n_noise_dim)}noisedim_unbalanced.ob"
	)
	with open(file_path, 'rb') as f:
		r = pickle.load(f)
	df=r['df']
	feat_start = 3
	train_features = df[df['Type']=="Training"].iloc[:,3:].values
	test_features = df[df['Type']=="Testing"].iloc[:,3:].values
	train_target = df[df['Type']=="Training"]['Class'].values
	test_target = df[df['Type']=="Testing"]['Class'].values
	train_cluster_labels = df[df['Type']=="Training"]['cluster_labels'].values
	test_cluster_labels = df[df['Type']=="Testing"]['cluster_labels'].values
	Fweights_train = r['Fweights_train']
	Fweights_test = r['Fweights_test']
	return train_features, test_features, train_target, test_target, train_cluster_labels, test_cluster_labels, Fweights_train, Fweights_test


def file_updater(file_path, rows, mode='a'):
	with open(file_path, mode, newline='', encoding='utf-8') as f:
		writer=csv.writer(f)
		for row in rows:
			writer.writerow(row)

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
													 monitor='val_auc',
													 mode='max',
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
		"n_attention": 3,
		"n_attention_hidden": 64, #512,
		'n_feat': None,
		"n_out": 1,
		"n_concat_hidden": 64, #512,
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
		"n_hidden": 64, #512,
	#     "hidden_activation": "selu", 
		"hidden_activation": tf.keras.layers.LeakyReLU(alpha=0.1),
		"kernel_regularizer": l2(1E-5),
		"bias_regularizer": l2(1E-5),
		"output_activation": "sigmoid",
		"random_seed": 123
	}
	return model_setup_params


def get_model_compile_params(learning_rate):
	model_compile_params = {
		"optimizer": tf.keras.optimizers.Adadelta(learning_rate=learning_rate,
												  clipvalue=0.5,
												  clipnorm=1.0
												  ),
		"loss": tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
		"metrics": ["AUC",]
	}
	return model_compile_params


def get_dataset_params():
	dataset_params = {
		"n_batch": 8,
		"n_buffer": 100,
	}
	return dataset_params


def get_model(model_type,
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
			   output_activation,
			   random_seed=123):
	tf.random.set_seed(random_seed)
	if model_type == "LSwFW" or model_type =="LSwFW_ones":
		input_shape = (n_feat*2,)
	else:
		input_shape = (n_feat,)
	input_layer = Input(shape=input_shape)

	if model_type == "dense":
		print(f"First dense layer with {n_attention*n_attention_hidden*n_attention_out} hidden unites")
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
	# dropout2 = Dropout(0.1)(dense_layer2)
	# # batchnorm2 = BatchNormalization(trainable=False,
	# # 								renorm=True
	# # 								)(dropout2)
	# dense_layer3 = Dense(n_hidden,
	# 					 activation=hidden_activation,
	# 					 kernel_initializer=kernel_initializer,
	# 					 kernel_regularizer=kernel_regularizer,
	# 					 bias_initializer=bias_initializer,
	# 					 bias_regularizer=bias_regularizer,
	# 					 )(dropout2)
	output_layer = Dense(n_out, activation=output_activation)(dense_layer2)

	tf_model = tf.keras.Model(inputs=input_layer,
							  outputs=output_layer
							  )
	return tf_model

def model_eval(predict, target, target_scaler, model_name, n_noise_dim, split_name="Train",
):
	if target_scaler is not None:
		predict = target_scaler.inverse_transform(
			predict
		)
	auc = roc_auc_score(target, predict.flatten())
	acc = accuracy_score(target, predict.flatten()>0.5)

	results=[
		[model_name, n_noise_dim, 'AUC', split_name, auc], 
		[model_name, n_noise_dim, 'acc', split_name, acc]
	]
	return results