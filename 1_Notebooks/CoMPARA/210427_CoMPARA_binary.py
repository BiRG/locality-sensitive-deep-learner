#!/usr/bin/env python3
# File name: 210426_AChEi_278cmpds.py

import os
import sys
import time
import csv

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import numpy as np

from CoMPARA_experiment_helper import load_data, data_scaling
from CoMPARA_experiment_helper import get_model_setup_params, get_model_compile_params, setup_callback_paths
from CoMPARA_experiment_helper import get_model
from CoMPARA_experiment_helper import file_updater, model_eval, model_score

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", dest="model_type", help="model_type: dense, LS, LSwFW, LSwFW_ones")
parser.add_argument("--resample", dest="resample", help="pass True to turn on minority class resampling in place of sample weights", default=False)

args = parser.parse_args()

if __name__ == "__main__":
    model_type=args.model_type
    resample = args.resample

    assert np.isin([model_type], ["dense", "LS", "LSwFW", "LSwFW_ones"])[0], "Choose model type as one of: 'dense', 'LS', 'LSwFW'."

    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.keras.backend.set_floatx('float32')

    # Load data
    r = load_data()
    labels = r['labels']
    train_features = r['train_features']
    test_features = r['test_features']
    train_targets = r['train_targets']
    test_targets = r['test_targets']
    Fweights_train = r['Fweights_train']
    Fweights_test = r['Fweights_test']

    # Setup summary file

    folder_name = "_".join([
        time.strftime("%y%m%d", time.localtime()),
        f"{model_type}",
        "lr0_0001"
        ])
    super_folder = os.path.join("CoMPARA_singlelabelmodels", folder_name)

    summary_csv_path = os.path.join(super_folder, "..", folder_name+".csv")
    header = ["Model", "Label", "Metric", "Split", "Score"]
    file_updater(summary_csv_path, [header], mode='w+')

    binary_labels = [0, 2, 4]

    for idx in binary_labels:
        #Get target
        label = labels[idx]
        valid_train_ind = np.where(~np.isnan(train_targets.iloc[:, idx]))[0]
        valid_test_ind = np.where(~np.isnan(test_targets.iloc[:, idx]))[0]

        train_target = train_targets.iloc[valid_train_ind, idx]
        test_target = test_targets.iloc[valid_test_ind, idx]

        # Scale features
        feature_scaler = StandardScaler()
        target_scaler = None
        train_features_scaled, test_features_scaled, y_train, y_test, feature_scaler, target_scaler = data_scaling(
            feature_scaler, target_scaler, train_features, test_features, train_target, test_target
        )
        if model_type=="LSwFW":
            X_train = np.hstack([train_features_scaled[valid_train_ind,:], Fweights_train[valid_train_ind,:]])
            X_test = np.hstack([test_features_scaled[valid_test_ind,:], Fweights_test[valid_test_ind,:]])
        elif model_type=="LSwFW_ones":
            X_train = np.hstack([train_features_scaled[valid_train_ind,:], 
                np.ones_like(train_features_scaled[valid_train_ind,:])])
            X_test = np.hstack([test_features_scaled[valid_test_ind,:], 
                np.ones_like(test_features_scaled[valid_test_ind,:])])
        else:
            X_train = train_features_scaled[valid_train_ind,:]
            X_test = test_features_scaled[valid_test_ind,:]

        # Get model parameters
        learning_rate = 0.0001
        model_setup_params = get_model_setup_params()
        model_compile_params = get_model_compile_params(learning_rate)
        n_feat = train_features_scaled.shape[1]
        model_setup_params['n_feat'] = n_feat
        model_setup_params['kernel_initializer']=tf.keras.initializers.HeNormal()

        # Setup callback paths
        callbacks, checkpoint_path = setup_callback_paths(
            "val_loss",
            mode="min",
            model_name=f"{model_type}_lr0_0001",
            dataset_name=f"CoMPARA_binary",
            split_number=f"{label}",
            super_folder=super_folder
        )

        if resample: #Resample minority classes
            from CoMPARA_experiment_helper import get_resampled_ds, get_test_ds
            n_batch, n_buffer = 16, 100
            #Set up resampled tf data
            train_ds, resampled_steps_per_epoch = get_resampled_ds(
                X_train, 
                y_train,
                n_batch=n_batch,
                n_buffer=n_buffer
            )
            test_ds = get_test_ds(
                X_test, 
                y_test, 
                n_batch,
                n_buffer
            )

        else: #Use class weights instead of resampling for unbalanced classes problem
            sample_weight = compute_sample_weight("balanced", y_train)

        # Setup model and train
        model = get_model(model_type, **model_setup_params)
        model.compile(**model_compile_params)

        print(f"Starting on {label} training")
        if resample:
            model.fit(
                train_ds, 
                validation_data = test_ds, 
                epochs=500, 
                steps_per_epoch = resampled_steps_per_epoch,
                callbacks=callbacks,
                verbose=2
            )
        else:
            model.fit(
                X_train,
                y_train,
                validation_data=(X_test,
                                 y_test),
                sample_weight=sample_weight,
                epochs=100,
                callbacks=callbacks,
                verbose=2,
            )

        # Evaluate model
        model.load_weights(checkpoint_path)
        # thresh=np.mean(y_train)
        results_train, sn_train, sp_train = model_eval(
            model,
            X_train,
            y_train.astype(int),
            target_scaler,
            model_name=model_type,
            label=label,
            split_name="Train"
        )
        results_test, sn_test, sp_test = model_eval(
            model,
            X_test,
            y_test.astype(int),
            target_scaler,
            model_name=model_type,
            label=label,
            split_name="Test"
        )
        results_modelscore = model_score(sn_train, sp_train, sn_test, sp_test,
                                         model_name=model_type,
                                         label=label,
                                         )
        file_updater(summary_csv_path, results_train, mode='a')
        file_updater(summary_csv_path, results_test, mode='a')
        file_updater(summary_csv_path, results_modelscore, mode='a')
    sys.exit()
