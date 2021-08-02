#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import time
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.regularizers import l1, l2

from Tox21experiment_helper import *

import sys
sys.path.append(os.path.join("..","..","0_code"))

if __name__ == "__main__":
    import csv

    labels, all_train_features, test_features, all_train_targets, test_targets, train_id_df, test_id_df, all_train_Fweights, test_Fweights = load_data()

    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.backend.set_floatx('float32')

    from algorithms import attention_model
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    from tf_helpers import get_weights_dicts, BinaryCrossEntropyIgnoreNan
    from tf_helpers import SimilarityBatchingDataset
    from tf_helpers import AveragedAUCIgnoreNan

    learning_rate = 0.001 #Fix this later
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

# ADD ARGPARSER for setup kwargs??
    model_setup_params = get_model_setup_params()
    weights_dicts = get_weights_dicts(all_train_targets)
    model_compile_params = get_model_compile_params(weights_dicts, learning_rate)
    dataset_params = get_dataset_params()
    n_batch=dataset_params['n_batch']
    n_buffer=dataset_params['n_buffer']

    params = {
    'n_attention_out':3,
    'n_attention': 10,
    }

    from sklearn.model_selection import KFold
    import time
    import dill as pickle
    #Get train-val split
    kf = KFold(
        n_splits = 10,
        shuffle = True,
        random_state = 1234 
        )

    for idx, (train_ind, val_ind) in enumerate(kf.split(all_train_features)):

        #Update Model setup parameters
        for key in params.keys():
            model_setup_params[key] = params[key]
        # split_name = "__".join([f"{key}_{params[key]}" for key in params.keys()])
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
        ds_dict = setup_tf_datasets(
            train_features_scaled, 
            train_targets_scaled,
            val_features_scaled,
            val_targets_scaled, 
            test_features_scaled,
            test_targets_scaled, 
            train_Fweights,
            val_Fweights,
            test_Fweights, 
            shuffle_buffer = n_buffer,
            batch_size = n_batch,
            )

        def load_ds_dict(ds_dict):
            train_ds = ds_dict.get('train_ds')
            val_ds = ds_dict.get('val_ds')
            test_ds = ds_dict.get('test_ds')
            return train_ds, val_ds, test_ds

        #Get callbacks and save paths
        callbacks, checkpoint_path = setup_callback_paths("val_averaged_auc_ignore_nan",
                                                          mode="max",
                                                          model_name="LSwFW_",
                                                          dataset_name="Tox21",
                                                          split_number=idx,
                                                          super_folder="SimilarityBatching"
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
        train_ds, val_ds, test_ds = load_ds_dict(ds_dict)

        for epoch in range(200):
            if epoch%2==1:
                train_tensor = np.hstack([train_features_scaled, train_Fweights])
                attentions = LSwFW_model.layers[1](train_tensor).numpy()
                simbatched = SimilarityBatchingDataset(
                    train_tensor,
                    train_targets_scaled,
                    attentions
                    )
                rearranged_train_data, rearranged_train_targets = simbatched.get_rearranged_tensor()
                ds_dict_rearranged = setup_tf_datasets(
                    rearranged_train_data, 
                    rearranged_train_targets, 
                    shuffle_buffer = n_buffer, 
                    batch_size = n_batch
                    )
                train_ds_rearranged, _, _ = load_ds_dict(ds_dict_rearranged)

                LSwFW_model.fit(train_ds_rearranged,
                                validation_data=val_ds,
                                epochs=1,
                                callbacks=callbacks,
                                verbose=2
                                )

            else:
                LSwFW_model.fit(train_ds,
                                validation_data=val_ds,
                                epochs=1,
                                callbacks=callbacks,
                                verbose=2
                                )

        # Evaluate model
        LSwFW_model.load_weights(checkpoint_path)
        LSwFW_model.save(os.path.join(checkpoint_path, "..", "saved_model"))

        test_tensor_scaled = np.hstack([test_features_scaled, test_Fweights])
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
        break
    sys.exit()