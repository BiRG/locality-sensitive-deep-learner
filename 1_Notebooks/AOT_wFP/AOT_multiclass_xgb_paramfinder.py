import os, sys
import pandas as pd
import numpy as np
import time
import csv
import dill as pickle

def get_cm(y_true, y_pred):
  ind=np.isfinite(y_true.astype(np.float))
  return confusion_matrix(y_true[ind].astype(np.int32), y_pred[ind])

def get_sn_sp(cm):
  tn, fp, fn, tp = cm.ravel()
  sn = np.float(tp)/(tp+fn)
  sp = np.float(tn)/(tn+fp)
  return sn, sp

def get_qual_model_score(sn_train, sp_train, sn_test, sp_test):
  ba_train = (sn_train+sp_train)/2.
  ba_test = (sn_test+sp_test)/2.
  gof = (0.7*ba_train) + 0.3*(1-np.abs(sn_train-sp_train))
  pred = (0.7*ba_test) + 0.3*(1-np.abs(sn_test-sp_test))
  rob = 1-np.abs(ba_train-ba_test)
  s = (0.3*gof) + (0.45*pred) + (0.25*rob)
  return s

def tune_xgbc(params, X_train, y_train, X_test, y_test):
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
    mdl = xgb.XGBClassifier(learning_rate = learning_rate, 
                            n_estimators = n_estimators, 
                            max_depth = max_depth, 
                            min_child_weight = min_child_weight, 
                            gamma = gamma, 
                            subsample = subsample, 
                            colsample_bytree = colsample_bytree, 
                            objective="multi:softprob",
                            seed = 42)

    sample_weight = compute_sample_weight("balanced", y_train)

    # #Cross-Validation in order to avoid overfitting
    # auc = cross_val_score(mdl, X_train_selected, y_train, cv = 10, scoring = 'roc_auc')

    mdl.fit(X_train, y_train, sample_weight = sample_weight)
    test_predict = mdl.predict_proba(X_test)
    test_auc = roc_auc_score(y_test, test_predict, multi_class = 'ovr')

    print(test_auc.mean())
    # as the function is minimization (forest_minimize), we need to use the negative of the desired metric (AUC)
    return -test_auc.mean()

if __name__ =="__main__":
	load_folder = os.path.join("data")
	scripts_folder = os.path.join("Scripts")
	sys.path.append(scripts_folder)

	from sklearn.preprocessing import StandardScaler, MinMaxScaler

	from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU
	from tensorflow.keras.initializers import VarianceScaling
	from tensorflow.keras.regularizers import l1, l2
	from tensorflow.keras.losses import BinaryCrossentropy

	from tf_helpers import get_weights_dicts, BinaryCrossEntropyIgnoreNan, AveragedAUCIgnoreNan
	from tf_helpers import SimilarityBatchingDataset

	from AOTexperiment_helper import load_data, setup_tf_datasets, data_scaling
	from AOTexperiment_helper import get_model_setup_params, get_model_compile_params, get_dataset_params, setup_callback_paths, setup_tf_datasets

	from sklearn.metrics import confusion_matrix

	import tensorflow as tf
	gpus = tf.config.experimental.list_physical_devices('GPU')
	for gpu in gpus:
	    tf.config.experimental.set_memory_growth(gpu, True)
	tf.keras.backend.set_floatx('float32')

	from tensorflow.keras.utils import to_categorical

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
	multiclass_labels = [3,4]
	train_features_scaled, test_features_scaled, train_targets_scaled, test_targets_scaled, feature_scaler, target_scaler = data_scaling(
		feature_scaler,
		target_scaler,
		train_features,
		test_features,
		train_targets[:, multiclass_labels].astype(np.float32),
		test_targets[:, multiclass_labels].astype(np.float32)
	)


	#Code adapted from Kaggle discussion board: https://www.kaggle.com/general/17120
	#Use random forest to optimize
	from skopt import forest_minimize
	from sklearn.utils.class_weight import compute_sample_weight
	from sklearn.metrics import roc_auc_score
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
	idx=0
	valid_train_ind = np.where(~np.isnan(train_targets_scaled[:, idx]))[0]
	valid_test_ind = np.where(~np.isnan(test_targets_scaled[:, idx]))[0]
	X_train=train_features_scaled[valid_train_ind]
	X_test = test_features_scaled[valid_test_ind]
	y_train=train_targets_scaled[valid_train_ind, idx]-1
	y_test = test_targets_scaled[valid_test_ind, idx]-1
	func=partial(tune_xgbc, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
	result = forest_minimize(func, space, random_state = 42, n_random_starts = 20, n_calls  = 25, verbose = 1)    

	with open("AOT_multiclass_xgb_params_optimizer.ob", 'wb') as f:
		pickle.dump(result, f)