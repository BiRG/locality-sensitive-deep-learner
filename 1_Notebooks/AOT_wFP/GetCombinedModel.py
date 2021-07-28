#!/usr/bin/env python
# coding: utf-8

# # Get results for combined model

# In[1]:


import os
import numpy as np
import pandas as pd

from AOTexperiment_helper import load_data, get_model, data_scaling, file_updater 
from AOTexperiment_helper import get_model_setup_params, get_model_compile_params#, setup_callback_paths
# from AOTexperiment_helper import cont_model_eval, get_cont_model_score
from AOTexperiment_helper import get_model_predictions

import time

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


# In[2]:


#Get scaled data
from sklearn.preprocessing import StandardScaler
feature_scaler = StandardScaler()
target_scaler = None
binary_labels = [0, 1]
train_features_scaled, test_features_scaled, train_targets_scaled, test_targets_scaled, feature_scaler, target_scaler = data_scaling(
    feature_scaler,
    target_scaler,
    train_features,
    test_features,
    train_targets,
    test_targets
)
n_feat = train_features_scaled.shape[1]

def get_valid_ind(y):
    ind = np.where(~np.isnan(y.astype(np.float32)))[0]
    return ind


from AOTexperiment_helper import get_cm, get_sn_sp, get_qual_model_score
def get_model_eval(y_train, y_test, train_predict, test_predict, num_classes=2):
    if num_classes>2:
        sn_train_list, sp_train_list = [],[]
        sn_test_list, sp_test_list=[],[]
        for i in range(num_classes):
            if len(np.unique(y_train[:, i]))<2:
                continue
            cm = get_cm(y_train[:,i], train_predict[:,i], )
            sn_train, sp_train = get_sn_sp(cm)
            cm = get_cm(y_test[:,i], test_predict[:,i], )
            sn_test, sp_test = get_sn_sp(cm)
            sn_train_list.append(sn_train)
            sp_train_list.append(sp_train)
            sn_test_list.append(sn_test)
            sp_test_list.append(sp_test)
        sn_train = np.mean(sn_train_list)
        sp_train = np.mean(sp_train_list)
        sn_test = np.mean(sn_test_list)
        sp_test = np.mean(sp_test_list)
    else:
        cm = get_cm(y_train, train_predict, num_classes=num_classes)
        sn_train, sp_train = get_sn_sp(cm)
        cm = get_cm(y_test, test_predict)
        sn_test, sp_test = get_sn_sp(cm)
    model_score = get_qual_model_score(sn_train, sp_train, sn_test, sp_test)
    return model_score, (sn_train, sp_train, sn_test, sp_test)
    

# In[42]:
# Get predictions for each model_type

#Update this for each model_type
model_types=["dense", "LS", "LSwFW", "LSwFW_ones", "xgboost"]
for model_type in model_types:
    folder_names = [
        f"210611_{model_type}_AOT_wFP_very_toxic_",
        f"210611_{model_type}_AOT_wFP_nontoxic_",
        f"210611_{model_type}_AOT_regression_LD50_mgkg",
        f"210614_{model_type}_AOT_wFP_EPA_category_", 
        f"210614_{model_type}_AOT_wFP_GHS_category_",
    ]

    if model_type == "xgboost":
        folder_names = [
            os.path.join("210614_xgboost_binary", "xgb_model_very_toxic"),
            os.path.join("210614_xgboost_binary", "xgb_model_nontoxic"),
            os.path.join("210614_xgboost_AOT_regression_LD50_mgkg", 
                         "xgb_model_LD50_mgkg"),
            os.path.join("210614_xgboost_multiclass", "xgb_model_EPA_category"),
            os.path.join("210614_xgboost_multiclass", "xgb_model_GHS_category"), 
        ]
    output_prefix = "_".join([model_type, "AOT"])

    n_class=[1,1,1,4,5]
    column_headers = [[f"{labels[i]}_{j}" for j in range(n_class[i])] for i in range(5)]
    column_headers = np.concatenate(column_headers)

    train_predicts = []
    test_predicts = []
    for label_idx in range(5):
        valid_train_ind = get_valid_ind(train_targets_scaled[:, label_idx])
        valid_test_ind = get_valid_ind(test_targets_scaled[:, label_idx])
        n_endpoints = None
        if label_idx < 2:
            endpoint = "binary"
        elif label_idx == 2:
            endpoint = "regression"
        else:
            endpoint = "multiclass"
            n_endpoints = len(np.unique(
                train_targets_scaled[valid_train_ind, label_idx]))

        checkpoint_path = os.path.join("AOT_models",
                                       folder_names[label_idx],
                                       "model_checkpoint"
                                       )
        X_train = train_features_scaled
        X_test = test_features_scaled
        if model_type=="xgboost":
            import xgboost as xgb
            checkpoint_path = os.path.join("AOT_models", 
                                           folder_names[label_idx]
                                          )
        elif model_type=="LSwFW":
            X_train = np.hstack([train_features_scaled,train_Fweights])
            X_test = np.hstack([test_features_scaled, test_Fweights])
        elif model_type =="LSwFW_ones":
            X_train = np.hstack([train_features_scaled, 
                                 np.ones_like(train_features_scaled)])
            X_test = np.hstack([test_features_scaled,
                                np.ones_like(test_features_scaled)
                               ])
        #Get predictions for all datapoints
        a, b = get_model_predictions(
            model_type,
            checkpoint_path,
            endpoint=endpoint,
            X_train=X_train,
            X_test=X_test,
            n_feat=n_feat,
            target_scaler=target_scaler,
            n_endpoints=n_endpoints
        )
        if a.ndim==1:
            a = a.reshape(-1,1)
            b = b.reshape(-1,1)
        train_predicts.append(a)
        test_predicts.append(b)

    train_predicts = pd.DataFrame(np.hstack(train_predicts),
                                  columns=column_headers)
    test_predicts = pd.DataFrame(np.hstack(test_predicts),
                                 columns=column_headers)
    train_predicts.to_csv("_".join([output_prefix,
                                    "train_predict"
                                    ]), index=False)
    test_predicts.to_csv("_".join([output_prefix,
                                   "test_predict"
                                   ]), index=False)


# ## Get combined predictions
combined_file = "_".join([
    time.strftime("%y%m%d", time.localtime()),
    "AOT",
    "combined_scores.csv"])
file_updater(combined_file, [["Model", "Label", "Metric", "Split", "Score"]], 
    mode="w")

for label_idx, label in enumerate(labels):
    date = "210611"
    l = label+"_"
    dataset = "AOT_wFP"
    if label_idx>2:
        date = "210614"
    if label_idx==2:
        l = label
        dataset = "AOT_regression"

    folder_names = [
        f"{date}_{model_types[0]}_{dataset}_{l}", 
        f"{date}_{model_types[1]}_{dataset}_{l}", 
        f"{date}_{model_types[2]}_{dataset}_{l}", 
        f"{date}_{model_types[3]}_{dataset}_{l}", 
    ]

# folder_names = [
#     "210611_dense_AOT_wFP_very_toxic_",
#     "210611_LS_AOT_wFP_very_toxic_",
#     "210611_LSwFW_AOT_wFP_very_toxic_",  
#     "210611_LSwFW_ones_AOT_wFP_very_toxic_"
# ]

# folder_names = [
#     "210517_dense_AOT_wFP_EPA_category_",
#     "210517_LS_AOT_wFP_EPA_category_",
#     "210517_LSwFW_AOT_wFP_EPA_category_", 
#     "210517_LSwFW_ones_AOT_wFP_EPA_category_"
# ]

# folder_names = [
#     "210515_dense_AOT_regression_LD50_mgkg",
#     "210515_LS_AOT_regression_LD50_mgkg",
#     "210516_LSwFW_AOT_regression_LD50_mgkg", 
#     "210516_LSwFW_ones_AOT_regression_LD50_mgkg"
# ]

# label_idx = 3
    model_types = ["dense", "LS", "LSwFW", "LSwFW_ones"]


    # In[17]:


    from tensorflow.keras.losses import CategoricalCrossentropy
    from tensorflow.keras.utils import to_categorical
    # Get predictions for each model_type
    train_predicts = []
    test_predicts = []
    for idx, model_type in enumerate(model_types):
        valid_train_ind = get_valid_ind(train_targets_scaled[:, label_idx])
        valid_test_ind = get_valid_ind(test_targets_scaled[:, label_idx])
        n_endpoints = None
        if label_idx < 2:
            endpoint = "binary"
        elif label_idx == 2:
            endpoint = "regression"
        else:
            endpoint = "multiclass"
            n_endpoints = len(np.unique(
                train_targets_scaled[valid_train_ind, label_idx]))

        checkpoint_path = os.path.join("AOT_models",
                                       folder_names[idx],
                                       "model_checkpoint"
                                       )
        if model_type=="LSwFW":
            X_train = np.hstack([train_features_scaled[valid_train_ind],
                                 train_Fweights[valid_train_ind]
                                ])
            X_test = np.hstack([test_features_scaled[valid_test_ind], 
                                test_Fweights[valid_test_ind]
                               ])
        elif model_type =="LSwFW_ones":
            X_train = np.hstack([
                train_features_scaled[valid_train_ind], 
                np.ones_like(train_Fweights[valid_train_ind])
                                ])
            X_test = np.hstack([
                test_features_scaled[valid_test_ind], 
                np.ones_like(test_Fweights[valid_test_ind])
            ])
        else:
            X_train = train_features_scaled[valid_train_ind]
            X_test = test_features_scaled[valid_test_ind]

        a, b = get_model_predictions(
            model_type,
            checkpoint_path,
            endpoint=endpoint,
            X_train=X_train,
            X_test=X_test,
            n_feat=n_feat,
            target_scaler=target_scaler,
            n_endpoints=n_endpoints
        )
    #     if endpoint =="multiclass":
    #         a = np.expand_dims(np.argmax(a, axis=1)+1, axis=1)
    #         b = np.expand_dims(np.argmax(b, axis=1)+1, axis=1)
        train_predicts.append(a)
        test_predicts.append(b)

    train_predicts = pd.DataFrame(np.hstack(train_predicts),
                                 )#columns=model_types)
    test_predicts = pd.DataFrame(np.hstack(test_predicts),
                                )#columns=model_types)
    # train_predicts.to_csv("_".join([output_prefix,
    #                                 "train_predict"
    #                                 ]), index=False)
    # test_predicts.to_csv("_".join([output_prefix,
    #                                "test_predict"
    #                                ]), index=False)


# In[18]:


# In[22]:
    combined_name = ["Combined_LS", "Combined_LSwFW_COSA", "Combined_LSwFW_naive"]

    for combined_idx, model_ind in enumerate([(0,1), (0,2), (0,3)]):
        # model_ind=[0,3]

        y_train = train_targets_scaled[valid_train_ind, label_idx]
        y_test = test_targets_scaled[valid_test_ind, label_idx]
        combined_train_predict = np.mean(
            train_predicts.values[:,model_ind], axis=1)
        combined_test_predict = np.mean(test_predicts.values[:,model_ind],
                               axis=1)
        if endpoint=="binary":
            r=get_model_eval(y_train.astype(int), y_test.astype(int), 
                       train_predict=combined_train_predict>0.5,
                       test_predict=combined_test_predict>0.5,
                      )
            rows=[
                [combined_name[combined_idx], label, "Sensitivity", "Train", r[1][0]],
                [combined_name[combined_idx], label, "Specificity", "Train", r[1][1]],
                [combined_name[combined_idx], label, "Sensitivity", "Test", r[1][2]],
                [combined_name[combined_idx], label, "Specificity", "Test", r[1][3]],
                [combined_name[combined_idx], label, "Model Score", "traintest", r[0]],
            ]
            file_updater(combined_file, rows)
            # print(r)
        elif endpoint=="multiclass":
            combined_train_predict = [train_predicts.values[:, 
                    i*n_endpoints:(i+1)*n_endpoints]
                                     for i in model_ind]
            combined_train_predict = to_categorical(np.argmax(
                np.mean(combined_train_predict, axis=0),
                axis=1
            ), num_classes=n_endpoints)
            combined_test_predict = [test_predicts.values[:, 
                    i*n_endpoints:(i+1)*n_endpoints] 
                                     for i in model_ind]
            combined_test_predict = to_categorical(np.argmax(
                np.mean(combined_test_predict, axis=0), 
                axis=1
            ), num_classes=n_endpoints)
            y_train = to_categorical(y_train.astype(int)-1)
            y_test = to_categorical(y_test.astype(int)-1)
            r=get_model_eval(y_train, y_test, 
                             train_predict=combined_train_predict,
                             test_predict=combined_test_predict,
                             num_classes=n_endpoints
                            )    
            rows=[
                [combined_name[combined_idx], label, "Sensitivity", "Train", r[1][0]],
                [combined_name[combined_idx], label, "Specificity", "Train", r[1][1]],
                [combined_name[combined_idx], label, "Sensitivity", "Test", r[1][2]],
                [combined_name[combined_idx], label, "Specificity", "Test", r[1][3]],
                [combined_name[combined_idx], label, "Model Score", "traintest", r[0]],
            ]
            file_updater(combined_file, rows)            
            # print(r)
        else:
            y_train = np.log10(y_train.astype(np.float32))
            y_test = y_test.astype(np.float32)
            ind = np.where(y_test>10000.)[0]
            y_test[ind]=10000.
            y_test = np.log10(y_test)
            
            from sklearn.metrics import r2_score, mean_squared_error
            r2_train = r2_score(y_train, combined_train_predict)
            r2_test = r2_score(y_test, combined_test_predict)
            model_score = (0.3*r2_train) + (0.45*r2_test) +(0.25*(1-np.abs(r2_train-r2_test)))
            rows=[
                [combined_name[combined_idx], label, "R_squared", "Train", r2_train,],
                [combined_name[combined_idx], label, "R_squared", "Test", r2_test,],
                [combined_name[combined_idx], label, "Model Score", "traintest", model_score],
            ]
            file_updater(combined_file, rows)            
            # print(model_score, r2_train, r2_test)


