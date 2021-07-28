#!/usr/bin/env python3
# File name: AOTexperiment_helper.py
# Description: Helper functions for AOT experiments
# Author: Yap Xiu Huan
# Contact: yapxiuhuan@gmail.com
# Date: 05-04-2021 (dd-mm-yyyy)

import os
import pandas as pd
import numpy as np
from functools import partial
import tensorflow as tf
from logging import critical, error, info, warning, debug
import time
import csv

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.regularizers import l1, l2

from tf_helpers import BinaryCrossEntropyIgnoreNan, AveragedAUCIgnoreNan
import attention_model

from sklearn.metrics import r2_score, mean_squared_error

# Data loading
def load_data(load_folder, final_selected_features):
    dataset_name = "XY_AOT_CDKPaDEL_processed.csv"
    load_df = pd.read_csv(os.path.join(load_folder,
                                       dataset_name
                                       ),
                          )

    id_columns = range(7)
    # very_toxic, nontoxic, LD50_mgkg, EPA_category, GHS_category
    label_columns = range(7, 12)
    data_columns = range(12, load_df.shape[1])  # CDK+PaDEL descriptors

    training_ind = np.where(load_df['Type'] == 'Training')[0]
    testing_ind = np.where(load_df['Type'] == "Testing")[0]

    X_train = load_df.iloc[training_ind, data_columns]
    X_test = load_df.iloc[testing_ind, data_columns]
    y_train = load_df.iloc[training_ind, label_columns]
    y_test = load_df.iloc[testing_ind, label_columns]
    train_id_df = load_df.iloc[training_ind, id_columns]
    test_id_df = load_df.iloc[testing_ind, id_columns]

    # TBD: Load Fweights
    Fweights_train=pd.read_csv(os.path.join(load_folder,
                                         "AOT_CDKPaDEL_Fweights_train.csv"
                                        )).values
    Fweights_test=pd.read_csv(os.path.join(load_folder,
                                              "AOT_CDKPaDEL_Fweights_test.csv"
                                             )).values

    # Load selected features

    def median_imputation(X_train, X_test):
        for col in range(len(X_train[0])):
            med = np.nanmedian(X_train[:, col])

            train_nanind = np.where(np.isnan(X_train[:, col]))[0]
            test_nanind = np.where(np.isnan(X_test[:, col]))[0]

            if len(train_nanind) > 0:
                X_train[train_nanind, col] = med
            if len(test_nanind) > 0:
                X_test[test_nanind, col] = med
        return X_train, X_test

    # Load feature selection from Data Preprocessing steps
    selected_train_features = X_train.iloc[:, final_selected_features].values
    selected_test_features = X_test.iloc[:, final_selected_features].values

    selected_train_features, selected_test_features = median_imputation(
        selected_train_features, selected_test_features)

    ret_dict = {
        'selected_train_features': selected_train_features,
        'selected_test_features': selected_test_features,
        'train_targets': y_train,
        'test_targets': y_test,
        'labels': load_df.columns[label_columns],
        'train_id_df': train_id_df,
        'test_id_df': test_id_df,
        'train_Fweights': Fweights_train,
        'test_Fweights': Fweights_test,
    }
    return ret_dict


def setup_tf_datasets(
    train_features,
    train_targets,
    val_features=None,
    val_targets=None,
    test_features=None,
    test_targets=None,
    train_Fweights=None,
    val_Fweights=None,
    test_Fweights=None,
    shuffle_buffer=100,
    batch_size=8
):
    """Data should be processed and scaled prior to adding to tf pipeline.

    shuffle_buffer: n_buffer for shuffling. If set to False, tf.data.Dataset is not shuffled. 
    batch_size: n_batch_size for batching. If set to False, tf.data.Dataset is not batched. 
    If training attention model with feature weights, include train_Fweights and test_Fweights """
    
    __get_tf_ds = partial(_get_tf_ds, 
        shuffle_buffer=shuffle_buffer,
        batch_size=batch_size
        )
    ret_dict=dict()

    if train_Fweights is not None:
        train_features = np.hstack([train_features, train_Fweights])
        if val_features is not None:
            val_features = np.hstack([val_features, val_Fweights])
        if test_features is not None:
            test_features = np.hstack([test_features, test_Fweights])     
    ret_dict['train_ds'] = __get_tf_ds(train_features, train_targets, )
    if val_features is not None:
        ret_dict['val_ds'] = __get_tf_ds(val_features, val_targets,)
    if test_features is not None:
        ret_dict['test_ds'] = __get_tf_ds(test_features, test_targets)

    return ret_dict

def _get_tf_ds(X,y, shuffle_buffer = 100, batch_size = 8):
    y = tf.cast(y, dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle_buffer:
        ds = ds.shuffle(shuffle_buffer)
    if batch_size:
        ds = ds.batch(batch_size)
    return ds

# Model training
def get_attentionwFW_model(n_attention,
                           n_attention_hidden,
                           n_feat,
                           n_out,
                           n_concat_hidden,
                           concat_activation,
                           n_attention_out,
                           kernel_initializer,
                           bias_initializer,
                           attention_hidden_activation,
                           attention_output_activation,
                           n_hidden,
                           hidden_activation,
                           kernel_regularizer,
                           bias_regularizer,
                           output_activation,
                           random_seed=123
                           ):
    input_shape = (n_feat*2,)
    input_layer = Input(shape=input_shape)
    tf.random.set_seed(random_seed)
    attentions_layer = attention_model.ConcatAttentionswFeatWeights(
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
    # Removed dropout for attentions_layer because of Batch normalization
    dense_layer1 = Dense(n_hidden,
                         activation=hidden_activation,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_initializer=bias_initializer,
                         bias_regularizer=bias_regularizer,
                         )(attentions_layer)
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

    LS_model = tf.keras.Model(inputs=input_layer,
                              outputs=output_layer
                              )
    info("Initialized Locality sensitive model with Feature Weights")
    return LS_model


def get_attention_model(n_attention,
                        n_attention_hidden,
                        n_feat,
                        n_out,
                        n_concat_hidden,
                        concat_activation,
                        n_attention_out,
                        kernel_initializer,
                        bias_initializer,
                        attention_hidden_activation,
                        attention_output_activation,
                        n_hidden,
                        hidden_activation,
                        kernel_regularizer,
                        bias_regularizer,
                        output_activation,
                        random_seed=123
                        ):
    input_shape = (n_feat,)
    input_layer = Input(shape=input_shape)
    tf.random.set_seed(random_seed)
    attentions_layer = attention_model.ConcatAttentions(
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
    # Removed dropout for attentions_layer because of Batch normalization
    dense_layer1 = Dense(n_hidden,
                         activation=hidden_activation,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_initializer=bias_initializer,
                         bias_regularizer=bias_regularizer,
                         )(attentions_layer)
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

    LS_model = tf.keras.Model(inputs=input_layer,
                              outputs=output_layer
                              )
    info("Initialized Locality sensitive model with Feature Weights")
    return LS_model


def get_dense_model(n_attention,
                    n_attention_hidden,
                    n_feat,
                    n_out,
                    n_concat_hidden,
                    concat_activation,
                    n_attention_out,
                    kernel_initializer,
                    bias_initializer,
                    attention_hidden_activation,
                    attention_output_activation,
                    n_hidden,
                    hidden_activation,
                    kernel_regularizer,
                    bias_regularizer,
                    output_activation,
                    random_seed=123
                    ):
    input_shape = (n_feat,)
    input_layer = Input(shape=input_shape)
    tf.random.set_seed(random_seed)

    first_layer = Dense(
        n_attention*n_attention_hidden*4,
        activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
    )(input_layer)
    dense_layer0 = Dense(
        n_concat_hidden,
        activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer
    )(first_layer)

    # Removed dropout for attentions_layer because of Batch normalization
    dense_layer1 = Dense(n_hidden,
                         activation=hidden_activation,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_initializer=bias_initializer,
                         bias_regularizer=bias_regularizer,
                         )(dense_layer0)
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

    dense_model = tf.keras.Model(inputs=input_layer,
                                 outputs=output_layer
                                 )
    info("Initialized Dense model")
    return dense_model


def setup_callback_paths(monitor,
                         mode,
                         model_name,
                         dataset_name,
                         split_number=0,
                         super_folder=None,
                         ):
    if isinstance(split_number, (int, float)):
        split_name = "split{:02d}".format(split_number)
    else:
        split_name = split_number
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
            warning(error)
    checkpoint_path = os.path.join(save_folder,
                                   "model_checkpoint")
    csv_filename = os.path.join(checkpoint_path, "training_log.csv")
    try:
        os.mkdir(save_folder)
    except OSError as error:
        warning(error)
    try:
        os.mkdir(checkpoint_path)
    except OSError as error:
        warning(error)

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

    if train_target.ndim==1:
        train_target = np.expand_dims(train_target, axis=1)
        test_target = np.expand_dims(test_target, axis=1)

    if target_scaler is not None:
        target_scaler.fit(train_target)
        train_target_scaled = target_scaler.transform(
            train_target)
        test_target_scaled = target_scaler.transform(
            test_target)
    else:
        train_target_scaled = train_target
        test_target_scaled = test_target
    return train_features_scaled, test_features_scaled, train_target_scaled, test_target_scaled, feature_scaler, target_scaler


def get_model_setup_params(args_dict={}):
    model_setup_params = {
        "n_attention": 10,
        "n_attention_hidden": 40,
        'n_feat': None,
        "n_out": 1,
        "n_concat_hidden": 512,
        # "concat_activation": tf.keras.layers.LeakyReLU(alpha=0.1),
        "concat_activation": tf.nn.leaky_relu,
        "n_attention_out": 3,
        "kernel_initializer": tf.keras.initializers.Orthogonal(),
        # So that we have weights to train on each LeakyReLU neuron
        "bias_initializer": tf.keras.initializers.Constant(value=0.1),
        # "attention_hidden_activation": tf.keras.layers.LeakyReLU(alpha=0.1),
        "attention_hidden_activation": tf.nn.leaky_relu,
        "attention_output_activation": "sigmoid",
        "n_hidden": 512,
        # "hidden_activation": tf.keras.layers.LeakyReLU(alpha=0.1),
        "hidden_activation": tf.nn.leaky_relu,
        "kernel_regularizer": l2(1E-5),
        "bias_regularizer": l2(1E-5),
        "random_seed": 123
    }
    for key in args_dict.keys():
        set_attr(model_setup_params, key, args_dict[key])
    return model_setup_params


def get_model_compile_params(learning_rate, args_dict={}):
    model_compile_params = {
        "optimizer": tf.keras.optimizers.Adadelta(learning_rate=learning_rate,
                                                  clipvalue=0.5,
                                                  clipnorm=1.0
                                                  ),
        "loss": "mean_squared_error",
        # "loss": BinaryCrossEntropyIgnoreNan(weights_dicts=weights_dicts),
        "metrics": [  # "AUC", "acc",
            # AveragedAUCIgnoreNan(num_labels=2)
            tf.keras.metrics.RootMeanSquaredError(),
        ]
    }
    for key in args_dict.keys():
        set_attr(model_compile_params, key, args_dict[key])
    return model_compile_params


def get_dataset_params(args_dict={}):
    dataset_params = {
        "n_batch": 8,
        "n_buffer": 100,
    }
    for key in args_dict.keys():
        set_attr(dataset_params, key, args_dict[key])
    return dataset_params


def lr_scheduler(epoch, lr):
    if epoch < 20:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


# Loading and Evaluating models


def get_p(y):
    """Get prior probability of a label"""
    y = y[~np.isnan(y)]
    return np.sum(y)/len(y)


def load_tf_model(model_path,
                  checkpoint_path=None,
                  ):
    load_model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "AveragedAUCIgnoreNan": AveragedAUCIgnoreNan,
            "BinaryCrossEntropyIgnoreNan": BinaryCrossEntropyIgnoreNan,
        }
    )
    if checkpoint_path is not None:
        load_model.load_weights(checkpoint_path)
    return load_model


def csv_dump(file_path, rows):
    with open(file_path, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        for row in rows:
            writer.writerow(row)

# Model evaluation

def cont_model_eval(model, 
    X, y, target_scaler, model_name, label, split_name="Train",
):
    predict = np.array(model.predict(X))
    if target_scaler is not None:
        predict = target_scaler.inverse_transform(predict).flatten()
    r2 = r2_score(y, predict)
    rmse = mean_squared_error(y, predict, squared=False)

    results=[
        [model_name, label, 'Rsquared', split_name, r2], 
        [model_name, label, 'RMSE', split_name, rmse]

    ]
    return results, r2

def get_cm(y_true, y_pred):
  ind=np.isfinite(y_true.astype(np.float))
  return confusion_matrix(y_true[ind].astype(np.int32), y_pred[ind])

def get_sn_sp(cm):
  tn, fp, fn, tp = cm.ravel()
  sn = np.float(tp)/(tp+fn)
  sp = np.float(tn)/(tn+fp)
  return sn, sp

def get_cont_model_score(r2_train, r2_test, model_name, label):
  gof = r2_train
  pred = r2_test
  rob = 1-np.abs(r2_train-r2_test)
  s = (0.3*gof) + (0.45*pred) + (0.25*rob)
  results = [[model_name, label, "Model score", "TrainTest", s]]
  return results

def file_updater(file_path, rows, mode='a'):
    with open(file_path, mode, newline='', encoding='utf-8') as f:
        writer=csv.writer(f)
        for row in rows:
            writer.writerow(row)
import argparse