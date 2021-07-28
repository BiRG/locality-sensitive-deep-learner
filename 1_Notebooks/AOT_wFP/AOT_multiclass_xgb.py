import os, sys
import pandas as pd
import numpy as np
import time
import csv

from sklearn.metrics import confusion_matrix
import dill as pickle
from sklearn.preprocessing import StandardScaler


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

def file_updater(file_path, rows, mode='a'):
	with open(file_path, mode, newline='', encoding='utf-8') as f:
		writer=csv.writer(f)
		for row in rows:
			writer.writerow(row)

if __name__ == "__main__":
	load_folder = "data"
	scripts_folder = "Scripts"
	sys.path.append(scripts_folder)

	from AOTexperiment_helper import load_data, data_scaling

	from sklearn.metrics import confusion_matrix
	from sklearn.utils.class_weight import compute_sample_weight
	from tensorflow.keras.utils import to_categorical

	import xgboost as xgb

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

	with open("AOT_multiclass_xgb_params_optimizer.ob", 'rb') as f:
		params=pickle.load(f)
	params= params['x']

	learning_rate = params[0] 
	n_estimators = params[1] 
	max_depth = params[2]
	min_child_weight = params[3]
	gamma = params[4]
	subsample = params[5]
	colsample_bytree = params[6]

	super_folder = "AOT_models"
	folder_name = "_".join([time.strftime("%y%m%d", time.localtime()),
							"xgboost",
							"multiclass"]
						   )
	summary_csv_path = os.path.join(super_folder,
									folder_name+"_summary.csv"
									)	
	header = ["Model", "Label", "Metric", "Type", "Score"]
	file_updater(summary_csv_path, [header], mode='w')
	try:
		os.mkdir(os.path.join(super_folder, folder_name))
	except:
		pass

	for idx, label_idx in enumerate(multiclass_labels):
		print(f"Starting on {labels[label_idx]}")
		valid_train_ind = np.where(~np.isnan(train_targets_scaled[:,idx]))[0]
		valid_test_ind = np.where(~np.isnan(test_targets_scaled[:, idx]))[0]

		y_train = train_targets_scaled[valid_train_ind, idx]
		y_test = test_targets_scaled[valid_test_ind, idx]    	

		train_sample_weight = compute_sample_weight("balanced", y_train)

		rgs = xgb.XGBClassifier(
			learning_rate = learning_rate, 
			n_estimators = n_estimators, 
			max_depth=max_depth,
			min_child_weight=min_child_weight,
			gamma=gamma,
			subsample=subsample,
			colsample_bytree=colsample_bytree,
			objective="multi:softprob",
			n_jobs=-1, 
			)
		rgs.fit(train_features_scaled[valid_train_ind], 
				y_train,
				sample_weight = train_sample_weight)
		rgs.save_model(os.path.join(super_folder, folder_name, f"xgb_model_{labels[label_idx]}"))

		r = []

		train_predict = rgs.predict(train_features_scaled[valid_train_ind])
		train_predict = to_categorical(train_predict)
		test_predict = rgs.predict(test_features_scaled[valid_test_ind], )
		test_predict = to_categorical(test_predict)

		categorical_y_train = to_categorical(y_train)

		diff = categorical_y_train.shape[1]-train_predict.shape[1]
		if diff>0:
		  train_predict = np.hstack([train_predict, np.zeros((train_predict.shape[0], diff))])
		diff = categorical_y_train.shape[1]-test_predict.shape[1]
		if diff>0:
		  test_predict=np.hstack([test_predict,np.zeros((test_predict.shape[0], diff))])

		cm_train = get_cm(categorical_y_train, train_predict)

		sn_train, sp_train = get_sn_sp(cm_train)
		print(f"Label {labels[label_idx]}: train NER={np.mean([sn_train, sp_train]):.3f}, train Sensitivity={sn_train:.3f}, train Specificity={sp_train:.3f}.")
		r.append(["xgboost", labels[label_idx], "NER", "Train", np.mean([sn_train, sp_train])])
		r.append(["xgboost", labels[label_idx], "Sensitivity", "Train", sn_train])
		r.append(["xgboost", labels[label_idx], "Specificity", "Train", sp_train])

		cm_test = get_cm(to_categorical(y_test), test_predict)

		sn_test, sp_test = get_sn_sp(cm_test)
		print(f"Label {labels[label_idx]}: test NER={np.mean([sn_test, sp_test]):.3f}, test Sensitivity={sn_test:.3f}, test Specificity={sp_test:.3f}.")
		r.append(["xgboost", labels[label_idx], "NER", "Test", np.mean([sn_test, sp_test])])
		r.append(["xgboost", labels[label_idx], "Sensitivity", "Test", sn_test])
		r.append(["xgboost", labels[label_idx], "Specificity", "Test", sp_test])

		model_score = get_qual_model_score(sn_train, sp_train, sn_test, sp_test)
		r.append(["xgboost", labels[label_idx], "model score", "TrainTest", model_score])
		print(f"Label{labels[label_idx]}: model score={model_score:.3f}")

		file_updater(summary_csv_path, r, mode='a')
	sys.exit()