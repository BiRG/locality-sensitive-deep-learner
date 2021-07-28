#!/usr/bin/env python3
# File name: 210426_AChEi_278cmpds.py

import os
import sys
import time

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np

from AChEi_experiment_helper import load_data, data_scaling, get_model_setup_params, get_model_compile_params, get_dataset_params, setup_callback_paths, get_dense_model, get_attention_model, get_attentionwFW_model, file_updater, model_eval

if __name__ == "__main__":
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.keras.backend.set_floatx('float32')

    # Load data
    r = load_data()
    selected_features = r['selected_features']
    all_target = r['all_target']

    # Setup summary file
    folder_name = time.strftime("%y%m%d_dense_models", time.localtime())
    super_folder = os.path.join("AChEi_humanOnly_models", folder_name)
    try:
        os.mkdir(super_folder)
    except:
        pass
    summary_csv_path = os.path.join(super_folder, folder_name+".csv")
    header = ["Model", "Fold", "Metric", "Split", "Score"]
    file_updater(summary_csv_path, [header], mode='w+')

    dataset_params = get_dataset_params()
    n_buffer = dataset_params['n_buffer']
    n_batch = dataset_params['n_batch']

    # Generate splits
    rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1234)
    for idx, (train_ind, test_ind) in enumerate(rkf.split(selected_features)):
        # Split features
        train_features = selected_features.iloc[train_ind, :].values
        test_features = selected_features.iloc[test_ind, :].values
        train_target = all_target.values[train_ind]
        test_target = all_target.values[test_ind]

        # Scale features
        feature_scaler = StandardScaler()
        target_scaler = RobustScaler()
        train_features_scaled, test_features_scaled, train_target_scaled, test_target_scaled, feature_scaler, target_scaler = data_scaling(
            feature_scaler, target_scaler, train_features, test_features, train_target, test_target
        )

        # Setup tf datasets
        train_ds = tf.data.Dataset.from_tensor_slices((train_features_scaled,
                                                       train_target_scaled
                                                       ))
        test_ds = tf.data.Dataset.from_tensor_slices((test_features_scaled,
                                                      test_target_scaled
                                                      ))
        train_ds = train_ds.shuffle(n_buffer).batch(n_batch)
        test_ds = test_ds.shuffle(n_buffer).batch(n_batch)

        # Get model parameters
        learning_rate = 0.05
        model_setup_params = get_model_setup_params()
        model_compile_params = get_model_compile_params(learning_rate)
        n_feat = train_features_scaled.shape[1]
        model_setup_params['n_feat'] = n_feat

        # Setup callback paths
        callbacks, checkpoint_path = setup_callback_paths(
            "val_root_mean_square_error",
            mode="min",
            model_name="Dense_RobustScaler",
            dataset_name="AChEi_278cmpds",
            split_number=idx,
            super_folder=super_folder
        )

        # Setup model and train
        dense_model = get_dense_model(**model_setup_params)
        dense_model.compile(**model_compile_params)
        dense_model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=200,
            callbacks=callbacks,
            verbose=0
        )

        # Evaluate model
        dense_model.load_weights(checkpoint_path)
        results_train = model_eval(
            dense_model,
            train_features_scaled,
            train_target,
            target_scaler,
            model_name="Dense",
            fold_idx=idx,
            split_name="Train"
        )
        results_test = model_eval(
            dense_model,
            test_features_scaled,
            test_target,
            target_scaler,
            model_name="Dense",
            fold_idx=idx,
            split_name="Test"
        )
        file_updater(summary_csv_path, results_train, mode='a')
        file_updater(summary_csv_path, results_test, mode='a')

    sys.exit()
