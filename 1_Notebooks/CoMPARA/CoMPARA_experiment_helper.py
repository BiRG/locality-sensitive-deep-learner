#!/usr/bin/env python3
# File name: AChEi_experiment_helper.py

import time
from sklearn.metrics import confusion_matrix

import os, sys
sys.path.append(os.path.join("..", "..", "0_code"))
import pandas as pd

import tensorflow as tf
import numpy as np
from algorithms import attention_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.initializers import VarianceScaling, HeNormal, Orthogonal, Constant
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.random import set_seed
from tensorflow.keras import Model
from functools import partial 
import dill as pickle
import csv

from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_sample_weight

def load_data():
	df = pd.read_csv(os.path.join("data", "XY_CoMPARA_CDKPaDELFP_filtered2.csv"), index_col=0)
	
	label_start, feat_start = 2,8
	labels = df.columns[label_start:feat_start]
	train_features = df[df['Type']=="Training"].iloc[:, feat_start:]
	test_features = df[df['Type']=="Testing"].iloc[:, feat_start:]
	train_targets= df[df['Type']=="Training"].iloc[:, label_start:feat_start]
	test_targets = df[df['Type']=="Testing"].iloc[:, label_start:feat_start]

	Fweights_train = pd.read_csv(os.path.join("data", "XY_CoMPARA_CDKPaDELFP_filtered2_Fweights_train.csv")).values
	Fweights_test = pd.read_csv(os.path.join("data", "XY_CoMPARA_CDKPaDELFP_filtered2_Fweights_test.csv")).values

	r = {'labels': labels,
	'train_features': train_features,
	'test_features': test_features, 
	'train_targets': train_targets,
	'test_targets': test_targets,
	'Fweights_train': Fweights_train,
	'Fweights_test': Fweights_test,
	}
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

#Resampling using tf.data
#Find majority label (and designate as negative). All remaining labels are designated as positive
def get_resampled_ds(X, y, n_batch, n_buffer):
    uniques, counts = np.unique(y, return_counts=True)
    maj_label = uniques[np.argmax(counts)]
    maj_ind = np.where(y==maj_label)[0]
    min_ind = np.where(y!=maj_label)[0]
    
    neg_features, neg_labels = X[maj_ind], y[maj_ind]
    pos_features, pos_labels = X[min_ind], y[min_ind]
    
    resampled_ds, resampled_steps_per_epoch = _make_resampled_ds(
        pos_features, 
        pos_labels, 
        neg_features,
        neg_labels,
        n_batch, 
        n_buffer
    )

    return resampled_ds, resampled_steps_per_epoch

#Helper functions for resampling using tf.data 
def _make_ds(features, labels, n_buffer):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    ds = ds.shuffle(n_buffer).repeat()
    return ds

def _make_resampled_ds(pos_features, pos_labels, neg_features, neg_labels, n_batch, n_buffer):
    pos_ds = _make_ds(pos_features, pos_labels, n_buffer)
    neg_ds = _make_ds(neg_features, neg_labels, n_buffer)

    ds_weights = [1./len(pos_labels), 1./len(neg_labels)]
    ds_weights = ds_weights/np.sum(ds_weights)
    
    resampled_ds = tf.data.experimental.sample_from_datasets(
        [pos_ds, neg_ds], weights = ds_weights)
    resampled_ds = resampled_ds.batch(n_batch)    
    # a=1./ds_weights[1]
    a = 1./ds_weights[0]
    resampled_steps_per_epoch = 10*np.ceil(a*len(pos_labels)/n_batch)
    return resampled_ds, resampled_steps_per_epoch

def get_test_ds(X,y, n_batch, n_buffer):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(n_buffer).batch(n_batch)
    return ds

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

	cp_callback = ModelCheckpoint(filepath=checkpoint_path,
													 monitor=monitor,
													 mode=mode,
													 save_best_only=True,
													 save_weights_only=True,
													 verbose=1
													 )
	csvlogger_callback = CSVLogger(filename=csv_filename,
													  append=True
													  )
	return [cp_callback, csvlogger_callback], checkpoint_path

def get_model_setup_params():
	model_setup_params = {
		"n_attention": 10,
		"n_attention_hidden": 40, #512,
		'n_feat': None,
		"n_out": 1,
		"n_concat_hidden": 512,
		"concat_activation": LeakyReLU(alpha=0.1),
		"n_attention_out": 3,
	    "kernel_initializer": Orthogonal(),
		# "kernel_initializer": tf.keras.initializers.HeNormal(),
	#     "bias_initializer": tf.keras.initializers.Zeros(),
		"bias_initializer": Constant(value=0.1),     # So that we have weights to train on each LeakyReLU neuron
	    "attention_kernel_initializer": Orthogonal(),
		# "attention_kernel_initializer": tf.keras.initializers.HeNormal(),
	#     "attention_bias_initializer": tf.keras.initializers.Zeros(),
		"attention_bias_initializer": Constant(value=0.1),
		"attention_hidden_activation": LeakyReLU(alpha=0.1),
	#     "attention_hidden_activation": "selu",
		"attention_output_activation": "sigmoid", 
		"n_hidden": 512,
	#     "hidden_activation": "selu", 
		"hidden_activation": LeakyReLU(alpha=0.1),
		"kernel_regularizer": l2(1E-5),
		"bias_regularizer": l2(1E-5),
		"output_activation": "sigmoid",
		"random_seed": 123
	}
	return model_setup_params


def get_model_compile_params(learning_rate):
	model_compile_params = {
		"optimizer": Adam(learning_rate=learning_rate,
												  clipvalue=0.5,
												  clipnorm=1.0
												  ),
		"loss": BinaryCrossentropy(label_smoothing=0.1),
		"metrics": ["AUC"]
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
	set_seed(random_seed)
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
	batchnorm1 = BatchNormalization(trainable=False,
									renorm=True
									)(dropout1)
	dense_layer2 = Dense(n_hidden,
						 activation=hidden_activation,
						 kernel_initializer=kernel_initializer,
						 kernel_regularizer=kernel_regularizer,
						 bias_initializer=bias_initializer,
						 bias_regularizer=bias_regularizer
						 )(batchnorm1)
	dropout2 = Dropout(0.1)(dense_layer2)
	batchnorm2 = BatchNormalization(trainable=False,
									renorm=True
									)(dropout2)
	dense_layer3 = Dense(n_hidden,
						 activation=hidden_activation,
						 kernel_initializer=kernel_initializer,
						 kernel_regularizer=kernel_regularizer,
						 bias_initializer=bias_initializer,
						 bias_regularizer=bias_regularizer,
						 )(batchnorm2)
	output_layer = Dense(n_out, activation=output_activation)(dense_layer3)

	tf_model = Model(inputs=input_layer,
							  outputs=output_layer
							  )
	return tf_model

# def get_dense_model(**kwargs):
# 	model_type = "dense"
# 	return _get_model(model_type,
# 					  **kwargs
# 					  )


# def get_attention_model(**kwargs):
# 	model_type = "LS"
# 	return _get_model(model_type,
# 					  **kwargs)


# def get_attentionwFW_model(**kwargs):
# 	model_type = "LSwFW"
# 	return _get_model(model_type,
# 					  **kwargs
# 					  )

def get_cm(y_true, y_pred, num_classes=2):
  ind=np.isfinite(y_true.astype(np.float))
  return confusion_matrix(y_true[ind].astype(np.int32), y_pred[ind], labels=list(range(num_classes)))

def get_sn_sp(cm):
  tn, fp, fn, tp = cm.ravel()
  sn = np.float(tp)/(tp+fn)
  sp = np.float(tn)/(tn+fp)
  return sn, sp

def model_eval(model, 
	X, y, target_scaler, model_name, label, split_name="Train", num_classes=2
):
	"""if multiclass, then convert predict values to categorical"""
	predict = np.array(model(X))
	if target_scaler is not None:
		predict = target_scaler.inverse_transform(predict).flatten()

	predict = np.round(predict).astype(int)
	if num_classes >2:
		predict=to_categorical(predict-1, num_classes=num_classes)
		y = to_categorical(y-1, num_classes=num_classes)

		sn_list, sp_list = [],[]
		for i in range(num_classes):
			if len(np.unique(y[:,i]))<2: #Skip class if either neg or pos are missing
				continue
			cm = get_cm(y[:,i], predict[:,i])
			sn, sp = get_sn_sp(cm)
			sn_list.append(sn)
			sp_list.append(sp)
		sn, sp = np.mean(sn_list), np.mean(sp_list)
	else:
		predict=(predict>0.5).astype(np.int32).ravel()
		cm=get_cm(y, predict)
		sn, sp = get_sn_sp(cm)

	results=[
		[model_name, label, "NER", split_name, np.mean([sn, sp])],
		[model_name, label, 'Sensitivity', split_name, sn], 
		[model_name, label, 'Specificity', split_name, sp]
	]
	return results, sn, sp

def get_qual_model_score(sn_train, sp_train, sn_test, sp_test):
  ba_train = (sn_train+sp_train)/2.
  ba_test = (sn_test+sp_test)/2.
  gof = (0.7*ba_train) + 0.3*(1-np.abs(sn_train-sp_train))
  pred = (0.7*ba_test) + 0.3*(1-np.abs(sn_test-sp_test))
  rob = 1-np.abs(ba_train-ba_test)
  s = (0.3*gof) + (0.45*pred) + (0.25*rob)
  return s

def model_score(sn_train, sp_train, sn_test, sp_test, model_name, label, split_name="TrainTest"):
	model_score = get_qual_model_score(sn_train, sp_train, sn_test, sp_test)
	results = [
		[model_name, label, "Model Score", split_name, model_score]
	]
	return results


#Get model predictions
def get_model_predictions(model_type, 
                          checkpoint_path, 
                          endpoint, 
                          X_train, 
                          X_test, 
                          n_feat,
                          target_scaler = None, 
                          n_endpoints=None):
    """'endpoint' is one of 'binary', 'multiclass' or 'regression'."""
    if model_type=="xgboost":
        import xgboost as xgb
        model = xgb.Booster({'nthread':4})
        model.load_model(checkpoint_path)

        X_train = xgb.DMatrix(X_train)
        X_test = xgb.DMatrix(X_test)

    else: #Tensorflow model
        model_setup_params = get_model_setup_params()
        model_setup_params['n_out']=1
        model_setup_params['n_feat'] = n_feat 
        model_compile_params = get_model_compile_params(learning_rate = 0.0001)
        if endpoint=="multiclass":
            assert n_endpoints is not None, "n_endpoints cannot be None for model_type==multiclass"
            model_setup_params['n_out']=n_endpoints
            model_compile_params['loss']=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=0.1)
        elif endpoint=="regression":
            model_setup_params['output_activation']="linear"
            model_compile_params['loss'] = tf.keras.losses.MeanSquaredError()
            model_compile_params['metrics'] = [
                tf.keras.metrics.RootMeanSquaredError()]        

        model = get_model(model_type = model_type, **model_setup_params)
        model.compile(**model_compile_params)
        model.load_weights(checkpoint_path)
    
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    
    if target_scaler is not None:
        train_predict = target_scaler.inverse_transform(train_predict)
        test_predict = target_scaler.inverse_transform(test_predict)
    return train_predict, test_predict