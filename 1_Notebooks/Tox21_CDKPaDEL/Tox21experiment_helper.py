#!/usr/bin/env python3
# File name: Tox21experiment_helper.py
# Description: Helper functions for Tox21 experiments
# Author: Yap Xiu Huan
# Contact: yapxiuhuan@gmail.com
# Date: 02-04-2021 (dd-mm-yyyy)

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import os
import sys
import pandas as pd
import numpy as np
from logging import critical, error, info, warning, debug
import time
from functools import partial
import dill as pickle

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.regularizers import l1, l2

sys.path.append(os.path.join("..", "..", "0_code"))
from algorithms import attention_model
from tf_helpers import AveragedAUCIgnoreNan, BinaryCrossEntropyIgnoreNan

# Loading datasets
def load_data(load_all_features=True, load_v1_dataset = False):
    """Set load_all_features as False to use selected feat in to partition dataset"""
    dataset_path = os.path.join("processed_data",
                                "XY_Tox21_CDKPaDEL_processedFiltered.csv"
                                ) #This dataset has 500+ features
    if load_v1_dataset:
        dataset_path = os.path.join("processed_data", 
            "XY_Tox21_CDKPadel_processedFiltered_old.csv"
            ) #This dataset has 945 features
        Fweights_df = pd.read_csv(os.path.join("processed_data", 
            "Tox21_CDKPaDEL_Fweights_train_old.csv"
            ))
        Fweights_OOS_df = pd.read_csv(os.path.join("processed_data", 
            "Tox21_CDKPaDEL_Fweights_test_old.csv"
            ))
    else:
        assert load_all_features, "V2 dataset is already filtered, set load_all_features to True"
        Fweights_df = pd.read_csv(os.path.join("processed_data", 
            "XY_Tox21_CDKPaDEL_processedfiltered_Fweights_train.csv"
            ))
        all_train_Fweights=Fweights_df.values
        Fweights_OOS_df = pd.read_csv(os.path.join("processed_data", 
            "XY_Tox21_CDKPaDEL_processedfiltered_Fweights_test.csv"
            ))
        test_Fweights=Fweights_OOS_df.values
    load_df = pd.read_csv(dataset_path, index_col=0)

    label_columns = range(5, 17)  # 12 Tox21 labels
    labels = load_df.columns[label_columns].values
    data_columns = range(17, load_df.shape[1])
    id_columns = range(5)

    training_ind = np.where(
        np.isin(load_df['Type'], ['Training', 'Testing']))[0]
    testing_ind = np.where(load_df['Type'] == "Score")[0]
    info(f"Tox21 dataset has data type 'Training', 'Testing' and 'Score'. Training and Testing (leaderboard) data are reflected in train while Score (Final Evaluation) data is in test.")

    all_train_features = load_df.iloc[training_ind, data_columns].values
    test_features = load_df.iloc[testing_ind, data_columns].values
    all_train_targets = load_df.iloc[training_ind, label_columns].values
    test_targets = load_df.iloc[testing_ind, label_columns].values
    train_id_df = load_df.iloc[training_ind, id_columns]
    test_id_df = load_df.iloc[testing_ind, id_columns]

    if load_v1_dataset:
    # # Load Fweights (COSA algorithm should be re-runned with only the desired features)
    # Fweights_df = pd.read_csv(os.path.join("processed_data",
    #                                        "Tox21_CDKPaDEL_Fweights_train.csv"
    #                                        ))
        all_train_Fweights = Fweights_df[load_df.columns[data_columns]].values

    # Fweights_OOS_df = pd.read_csv(os.path.join("processed_data",
    #                                            "Tox21_CDKPaDEL_Fweights_test.csv"
    #                                            ))
        test_Fweights = Fweights_OOS_df[load_df.columns[data_columns]].values

    #Load selected features
    if not load_all_features:
        assert load_v1_dataset, "`load_all_features=False` only for v1 dataset. Feature selection is already applied in V2 dataset."
        with open("selected_feat_ind.ob", 'rb') as f:
            selected_feat_ind=pickle.load(f)

        all_train_features = all_train_features[:, selected_feat_ind]
        test_features = test_features[:, selected_feat_ind]
        all_train_Fweights = all_train_Fweights[:, selected_feat_ind]
        test_Fweights = test_Fweights[:, selected_feat_ind]

    return labels, all_train_features, test_features, all_train_targets, test_targets, \
        train_id_df, test_id_df, all_train_Fweights, test_Fweights


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

# Training models

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
                           random_seed=123
                           ):
    input_shape = (n_feat*2,)
    input_layer = Input(shape=input_shape)
    tf.random.set_seed(random_seed)
    if type(n_hidden) is int:
        n_hidden = [n_hidden] *3
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
    dense_layer1 = Dense(n_hidden[0],
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
    dense_layer2 = Dense(n_hidden[1],
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
    dense_layer3 = Dense(n_hidden[2],
                         activation=hidden_activation,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_initializer=bias_initializer,
                         bias_regularizer=bias_regularizer,
                         )(batchnorm2)
    output_layer = Dense(n_out, activation="sigmoid")(dense_layer3)

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
                        random_seed=123
                        ):
    input_shape = (n_feat,)
    input_layer = Input(shape=input_shape)
    tf.random.set_seed(random_seed)
    if type(n_hidden) is int:
        n_hidden = [n_hidden] *3
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
    dense_layer1 = Dense(n_hidden[0],
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
    dense_layer2 = Dense(n_hidden[1],
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
    dense_layer3 = Dense(n_hidden[2],
                         activation=hidden_activation,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_initializer=bias_initializer,
                         bias_regularizer=bias_regularizer,
                         )(batchnorm2)
    output_layer = Dense(n_out, activation="sigmoid")(dense_layer3)

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
                    random_seed=123
                    ):
    input_shape = (n_feat,)
    input_layer = Input(shape=input_shape)
    tf.random.set_seed(random_seed)
    if type(n_hidden) is int:
        n_hidden = [n_hidden]*3
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
    dense_layer1 = Dense(n_hidden[0],
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
    dense_layer2 = Dense(n_hidden[1],
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
    dense_layer3 = Dense(n_hidden[2],
                         activation=hidden_activation,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_initializer=bias_initializer,
                         bias_regularizer=bias_regularizer,
                         )(batchnorm2)
    output_layer = Dense(n_out, activation="sigmoid")(dense_layer3)

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


def get_model_setup_params(args_dict={}):
    model_setup_params = {
        "n_attention": 10,
        "n_attention_hidden": 40,
        'n_feat': None,
        "n_out": 12,
        "n_concat_hidden": 512,
        # "concat_activation": tf.keras.layers.LeakyReLU(alpha=0.1),
        "concat_activation":"selu", 
        "n_attention_out": 3,
        # "kernel_initializer": tf.keras.initializers.Orthogonal(),
        "kernel_initializer": tf.keras.initializers.VarianceScaling(),
        # So that we have weights to train on each LeakyReLU neuron
        "bias_initializer": tf.keras.initializers.Constant(value=0.1),
        # "attention_hidden_activation": tf.keras.layers.LeakyReLU(alpha=0.1),
        "attention_hidden_activation": "selu", 
        "attention_output_activation": "sigmoid",
        "n_hidden": 2048,
        # "hidden_activation": tf.keras.layers.LeakyReLU(alpha=0.1),
        "hidden_activation": "selu",
        "kernel_regularizer": l2(1E-5),
        "bias_regularizer": l2(1E-5),
        "random_seed": 123
    }
    for key in args_dict.keys():
        set_attr(model_setup_params, key, args_dict[key])
    return model_setup_params


def get_model_compile_params(weights_dicts, learning_rate, args_dict={}):
    model_compile_params = {
        "optimizer": tf.keras.optimizers.Adadelta(learning_rate=learning_rate,
                                                  clipvalue=0.5,
                                                  clipnorm=1.0
                                                  ),
        "loss": BinaryCrossEntropyIgnoreNan(weights_dicts=weights_dicts),
        "metrics": [  # "AUC", "acc",
            AveragedAUCIgnoreNan(num_labels=12)
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


def evaluate_Tox21model(tf_model,
                        X,
                        Y,
                        split="N/A",
                        labels=None,
                        target_scaler=None,
                        predict_thresh='a_priori',
                        Y_train=None,
                        write_to_file_path=None
                        ):
    """
    Evaluation for Tox21 tensorflow model

    predict_thresh (str, float): 'a_priori' or float between [0.0, 1.0].
    Setting this as 'a_priori' will adjust threshold classification score
    based on training a priori score `Y_train`. 
    """

    header = ["Model", "Split", "Label", "Metric", "Score"]
    if write_to_file_path is not None:
        csv_dump(write_to_file_path, [header])

    predict = tf_model(X).numpy()
    if target_scaler is not None:
        predict = target_scaler.inverse_transform(predict).numpy()

    results = []
    for label_idx in range(Y.shape[1]):
        if labels is not None:
            label_name = labels[label_idx]
        else:
            label_name = "Label{%s}".format(label_idx)
        ind = np.where(~tf.math.is_nan(Y[:, label_idx]))[0]
        if predict_thresh == 'a_priori':
            thresh = get_p(Y_train[:, label_idx])
        else:
            thresh = predict_thresh
        predict_class = tf.cast(tf.math.greater(predict[ind, label_idx], thresh),
                                dtype=tf.int32
                                )

        auc = roc_auc_score(Y[ind, label_idx], predict[ind, label_idx])
        f1 = f1_score(Y[ind, label_idx], predict_class)
        acc = accuracy_score(Y[ind, label_idx], predict_class)

        info(f"{label_name}: AUC = {auc:.3f}; f1 = {f1:.3f}; acc = {acc:.3f}.")

        results.append(["LS_model", split, label_name, "AUC", auc])
        results.append(["LS_model", split, label_name, "f1", f1])
        results.append(["LS_model", split, label_name, "acc", acc])

        if write_to_file_path is not None:
            csv_dump(write_to_file_path, results[-3:])

    return pd.DataFrame(results, columns=header)
