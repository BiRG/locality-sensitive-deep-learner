#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import time
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.regularizers import l1, l2

from Tox21experiment_helper import *


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
    model_compile_params = get_model_compile_params(weights_dicts, learning_rate)
    dataset_params = get_dataset_params()
    n_batch=dataset_params['n_batch']
    n_buffer=dataset_params['n_buffer']

    from sklearn.model_selection import RepeatedKFold, ParameterGrid
    import time
    import dill as pickle
    #Get train-val split
    rkf = RepeatedKFold(
        n_splits = 5,
        n_repeats = 10, 
        random_state = 1234 
        )
    for idx, (train_ind, val_ind) in enumerate(rkf.split(all_train_features)):
        break

    param_grid = {
        "n_attention": range(10, 12), 
        "n_attention_out": range(3, 6)
    }

    for params in ParameterGrid(param_grid):
        #Update Model setup parameters
        for key in params.keys():
            model_setup_params[key] = params[key]
        split_name = "__".join([f"{key}_{params[key]}" for key in params.keys()])
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
        
        # Load Feature weighting
        train_Fweights = all_train_Fweights[train_ind,:]
        val_Fweights = all_train_Fweights[val_ind,:]
        
        # Tensor casting
        train_targets_scaled = tf.cast(train_targets_scaled, tf.float32)
        val_targets_scaled = tf.cast(val_targets_scaled, tf.float32)
        test_targets_scaled = tf.cast(test_targets_scaled, tf.float32)
        
        # Setup tf datasets
        train_tensor_scaled = np.hstack([train_features_scaled, train_Fweights])
        val_tensor_scaled = np.hstack([val_features_scaled, val_Fweights])
        test_tensor_scaled = np.hstack([test_features_scaled, test_Fweights])
        train_ds = tf.data.Dataset.from_tensor_slices((train_tensor_scaled,
                                                       train_targets_scaled
                                                       ))
        val_ds = tf.data.Dataset.from_tensor_slices((val_tensor_scaled,
                                                     val_targets_scaled
                                                    ))
        test_ds = tf.data.Dataset.from_tensor_slices((test_tensor_scaled,
                                                      test_targets_scaled
                                                      ))
        train_ds = train_ds.shuffle(n_buffer).batch(n_batch)
        val_ds = val_ds.shuffle(n_buffer).batch(n_batch)
        test_ds = test_ds.shuffle(n_buffer).batch(n_batch)

        #Get callbacks and save paths
        callbacks, checkpoint_path = setup_callback_paths("val_averaged_auc_ignore_nan",
                                                          mode="max",
                                                          model_name="LSwFW_",
                                                          dataset_name="Tox21",
                                                          split_number=split_name,
                                                          super_folder="Parameter_finding"
                                                          )
        
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_scheduler))
        
        #Save run info
        to_save = {'train_ind': train_ind,
                   'val_ind': val_ind,
                   'feature_scaler': feature_scaler,
                   'target_scaler': target_scaler,
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
        LSwFW_model = get_attentionwFW_model(**model_setup_params
                                             )

        LSwFW_model.compile(**model_compile_params)

        # Fit and train model
        LSwFW_model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=100,
                        callbacks=callbacks,
                        verbose=2
                        )

        # Evaluate model
        LSwFW_model.load_weights(checkpoint_path)
        LSwFW_model.save(os.path.join(checkpoint_path, "..", "saved_model"))

        test_predict = LSwFW_model.predict(test_tensor_scaled)
        if target_scaler is not None:
            test_predict = target_scaler.inverse_transform(test_predict)

        from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
        AUCs = []
        for label_idx, label in enumerate(labels):
            ind = ~tf.math.is_nan(test_targets[:, label_idx])
            thresh = get_p(all_train_targets[:, idx])
            test_predict_class = tf.cast(tf.math.greater(test_predict[ind, label_idx], thresh),
                dtype = tf.int32)
            auc = roc_auc_score(test_targets[ind, label_idx], test_predict[ind, label_idx])
            f1 = f1_score(test_targets[ind, label_idx], test_predict_class)
            acc = accuracy_score(test_targets[ind, label_idx], test_predict_class)

            print(f"{label}: AUC = {auc:.3f}; f1 = {f1:.3f}; acc = {acc:.3f}.")

            with open(output_csv, 'a') as f:
                writer = csv.writer(f, delimiter = ',')
                writer.writerow(["LS_model", idx, label, "AUC", "Test", auc])
                writer.writerow(["LS_model", idx, label, "f1", "Test", f1])
                writer.writerow(["LS_model", idx, label, "acc", "Test", acc])

        
        del LSwFW_model

