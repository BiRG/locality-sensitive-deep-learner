#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import time
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.regularizers import l1, l2

from Tox21experiment_helper import *



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
        "concat_activation": tf.keras.layers.LeakyReLU(alpha=0.1),
        "n_attention_out": 3,
        "kernel_initializer": tf.keras.initializers.Orthogonal(),
        # So that we have weights to train on each LeakyReLU neuron
        "bias_initializer": tf.keras.initializers.Constant(value=0.1),
        "attention_hidden_activation": tf.keras.layers.LeakyReLU(alpha=0.1),
        "attention_output_activation": "sigmoid",
        "n_hidden": 2048,
        "hidden_activation": tf.keras.layers.LeakyReLU(alpha=0.1),
        "kernel_regularizer": l2(1E-5),
        "bias_regularizer": l2(1E-5),
        "random_seed": 123
    }
    for key in args_dict.keys():
        set_attr(model_setup_params, key, args_dict[key])
    return model_setup_params

def get_model_compile_params(weights_dicts, args_dict={}):
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
    if epoch <20:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def get_p(y):
    y=y[~np.is_nan(y)]
    return np.sum(y)/len(y)

if __name__ == "__main__":
    import sys
    import csv
    sys.path.append(os.path.join("..", "..", "0_code"))

    labels, all_train_features, test_features, all_train_targets, test_targets, train_id_df, test_id_df, all_train_Fweights, test_Fweights = load_data()

    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.backend.set_floatx('float32')

    from algorithms import attention_model
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    from tf_helpers import get_weights_dicts, BinaryCrossEntropyIgnoreNan
    # from tf_helpers import SimilarityBatchingDataset
    from tf_helpers import AveragedAUCIgnoreNan

    learning_rate = 0.005 #Fix this later
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

# ADD ARGPARSER for setup kwargs??
    model_setup_params = get_model_setup_params()
    weights_dicts = get_weights_dicts(all_train_targets)
    model_compile_params = get_model_compile_params(weights_dicts, )
    dataset_params = get_dataset_params()
    n_batch=dataset_params['n_batch']
    n_buffer=dataset_params['n_buffer']

    from sklearn.model_selection import RepeatedKFold, ParameterGrid
    import time
    import dill as pickle
    #Get train-val split
    rkf = RepeatedKFold(
        n_splits = 5,
        n_repeats = 5, 
        random_state = 1234 
        )
    for idx, (train_ind, val_ind) in enumerate(rkf.split(all_train_features)):
        n_feat = all_train_features.shape[1]
        model_setup_params['n_feat'] = n_feat   

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
        
        train_features_scaled = all_train_features_scaled[train_ind, :]
        val_features_scaled = all_train_features_scaled[val_ind, :]
        train_targets_scaled = all_train_targets_scaled[train_ind]
        val_targets_scaled = all_train_targets_scaled[val_ind]
        
        # Tensor casting
        train_targets_scaled = tf.cast(train_targets_scaled, tf.float32)
        val_targets_scaled = tf.cast(val_targets_scaled, tf.float32)
        test_targets_scaled = tf.cast(test_targets_scaled, tf.float32)
        
        # Setup tf datasets
        train_ds = tf.data.Dataset.from_tensor_slices((train_features_scaled,
                                                       train_targets_scaled
                                                       ))
        val_ds = tf.data.Dataset.from_tensor_slices((val_features_scaled,
                                                     val_targets_scaled
                                                    ))
        test_ds = tf.data.Dataset.from_tensor_slices((test_features_scaled,
                                                      test_targets_scaled
                                                      ))
        train_ds = train_ds.shuffle(n_buffer).batch(n_batch)
        val_ds = val_ds.shuffle(n_buffer).batch(n_batch)
        test_ds = test_ds.shuffle(n_buffer).batch(n_batch)

        #Get callbacks and save paths
        callbacks, checkpoint_path = setup_callback_paths("val_averaged_auc_ignore_nan",
                                                          mode="max",
                                                          model_name="dense_",
                                                          dataset_name="Tox21",
                                                          split_number=idx,
                                                          super_folder="Preliminary_Models"
                                                          )
        
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_scheduler))
        
        #Save run info
        to_save = {'train_ind': train_ind,
                   'val_ind': val_ind,
                   }
        with open(os.path.join(checkpoint_path, "..",
                               "input_info.ob"), 'wb') as f:
            pickle.dump(to_save, f)

        to_save = {
            "_call": get_attentionwFW_model,
            "model_setup_params": model_setup_params,
            "model_compile_params": model_compile_params,
            "dataset_params": dataset_params
        }
        with open(os.path.join(checkpoint_path, "..",
                               "model_params.ob"), 'wb') as f:
            pickle.dump(to_save, f)

        output_csv = os.path.join(checkpoint_path, "output.csv")
        
        header = ["Model", "Split", "Label", "Metric", "Type", "Score"]
        with open(output_csv, 'w+') as f:
            writer = csv.writer(f, delimiter = ',')
            writer.writerow(header)

        # Set up model
        dense_model = get_dense_model(**model_setup_params
                                             )

        dense_model.compile(**model_compile_params)

        # Fit and train model
        dense_model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=100,
                        callbacks=callbacks,
                        verbose=2
                        )

        # Evaluate model
        LSwFW_model.load_weights(checkpoint_path)
        test_predict = LSwFW_model.predict(test_tensor_scaled)
        if target_scaler is not None:
            test_predict = target_scaler.inverse_transform(test_predict)

        from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
        AUCs = []
        for label_idx, label in enumerate(labels):
            ind = ~tf.math.is_nan(test_targets[:, label_idx])
            thresh = get_p(all_train_targets[:, label_idx])
            test_predict_class = tf.cast(tf.math.greater(test_predict[ind, label_idx]), 
                dtype = tf.int32
                )
            auc = roc_auc_score(test_targets[ind, label_idx], test_predict[ind, label_idx])
            f1 = f1_score(test_targets[ind, label_idx], test_predict_class)
            acc = accuracy_score(test_targets[ind, label_idx], test_predict_class)

            print(f"{label}: AUC = {auc:.3f}; f1 = {f1:.3f}; acc = {acc:.3f}.")

            with open(output_csv, 'a') as f:
                writer = csv.writer(f, delimiter = ',')
                writer.writerow(["dense_model", idx, label, "AUC", "Test", auc])
                writer.writerow(["dense_model", idx, label, "f1", "Test", f1])
                writer.writerow(["dense_model", idx, label, "acc", "Test", acc])

        dense_model.save(os.path.join(checkpoint_path, "..", "saved_model"))
        
        del dense_model

