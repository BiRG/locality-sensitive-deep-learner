#!/usr/bin/env python3
# File name: 210503_AChEi_ExtendedDataset_allFeatures.py

import os
import sys
import time

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np

from AChEi_experiment_helper import load_data, data_scaling, get_model_setup_params, get_model_compile_params, get_dataset_params, setup_callback_paths, get_dense_model, get_attention_model, get_attentionwFW_model, file_updater, model_eval

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument(dest="model_type", help="model_type: dense, LS, LSwFW, LSwFW_ones, xgboost")
parser.add_argument('-i', dest="start_idx", default=0, help="idx to start at")
parser.add_argument('-humanOnly', dest='humanOnly', default=False, help="Whether to use human values only")
args = parser.parse_args()

if __name__ == "__main__":
    model_type=args.model_type
    start_idx = int(args.start_idx)
    humanOnly = bool(args.humanOnly)
    assert np.isin([model_type], ["dense", "LS", "LSwFW", "LSwFW_ones", "xgboost"])[0], "Choose model type as one of: 'dense', 'LS', 'LSwFW'."

    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.keras.backend.set_floatx('float32')

    # Load data
    r = load_data()
    # selected_features = r['selected_features']
    # all_target = r['all_target']
    valid_features = r['valid_features']
    valid_target = r['valid_target']
    valid_Fweights = r['valid_Fweights']

    if humanOnly:
        rows = np.where(valid_features['A'])[0]
        retain_col = range(7, len(valid_features.columns))
        valid_features = valid_features.iloc[:, retain_col].iloc[rows]
        valid_target = valid_target.iloc[rows]
        valid_Fweights = valid_Fweights[np.ix_(rows, retain_col)]

    if model_type=="LSwFW_ones":
        valid_Fweights = np.ones_like(valid_features)

    # Setup summary file
    folder_name = "_".join([time.strftime("%y%m%d", time.localtime()), model_type])
    super_folder = os.path.join("AChEi_extendedDataset_models", folder_name)
    if humanOnly: 
        super_folder = os.path.join("AChEi_humanOnly_models")
    try:
        os.mkdir(super_folder)
    except:
        pass
    summary_csv_path = os.path.join(super_folder, folder_name+".csv")
    if start_idx==0:
        header = ["Model", "Fold", "Metric", "Split", "Score"]
        file_updater(summary_csv_path, [header], mode='w+')

    # dataset_params = get_dataset_params()
    # n_buffer = dataset_params['n_buffer']
    # n_batch = dataset_params['n_batch']

    # Generate splits
    kf = KFold(n_splits=20, random_state=1234, shuffle=True)
    for idx, (train_ind, test_ind) in enumerate(kf.split(valid_features)):
        if start_idx>idx:
            continue

        # Split features
        train_features = valid_features.iloc[train_ind, :].values
        test_features = valid_features.iloc[test_ind, :].values
        train_target = valid_target.values[train_ind]
        test_target = valid_target.values[test_ind]
        train_Fweights = valid_Fweights[train_ind,:]
        test_Fweights = valid_Fweights[test_ind,:]

        # Scale features
        feature_scaler = StandardScaler()
        target_scaler = RobustScaler()
        train_features_scaled, test_features_scaled, train_target_scaled, test_target_scaled, feature_scaler, target_scaler = data_scaling(
            feature_scaler, target_scaler, train_features, test_features, train_target, test_target
        )

        if model_type=="xgboost":
            import xgboost as xgb
            model = xgb.XGBRegressor(
                n_estimators=499,
                objective = "reg:squarederror",
                n_jobs=2
                )    
            model.fit(train_features_scaled, train_target_scaled)

        # Get model parameters
        else:
            learning_rate = 0.5
            model_setup_params = get_model_setup_params()
            model_compile_params = get_model_compile_params(learning_rate)
            n_feat = train_features_scaled.shape[1]
            model_setup_params['n_feat'] = n_feat

            # Setup callback paths
            callbacks, checkpoint_path = setup_callback_paths(
                "val_root_mean_squared_error",
                mode="min",
                model_name=f"{model_type}_RobustScaler_tempNormKi",
                dataset_name="AChEi_extendedDataset_allFeatures",
                split_number=idx,
                super_folder=super_folder
            )

            # Setup model and train
            if model_type=="dense":
                model_call=get_dense_model
            elif model_type=="LS":
                model_call=get_attention_model
            elif model_type=="LSwFW" or model_type=="LSwFW_ones":
                model_call=get_attentionwFW_model
                train_features_scaled = np.hstack([train_features_scaled, train_Fweights])
                test_features_scaled = np.hstack([test_features_scaled, test_Fweights])
            model = model_call(**model_setup_params)
            model.compile(**model_compile_params)
            model.fit(
                train_features_scaled,
                train_target_scaled, 
                validation_data=(test_features_scaled, test_target_scaled),
                epochs=200,
                callbacks=callbacks,
                verbose=2
            )
            model.load_weights(checkpoint_path)

        # Evaluate model            
        results_train = model_eval(
            model,
            train_features_scaled,
            train_target,
            target_scaler,
            model_name=model_type,
            fold_idx=idx,
            split_name="FirstFit_Train"
        )
        results_test = model_eval(
            model,
            test_features_scaled,
            test_target,
            target_scaler,
            model_name=model_type,
            fold_idx=idx,
            split_name="FirstFit_Test"
        )
        file_updater(summary_csv_path, results_train, mode='a')
        file_updater(summary_csv_path, results_test, mode='a')
        if model_type=="xgboost":
            continue

        ##Reload last layer weights and train again
        weights = model.layers[-1].get_weights()
        weights[0] = model_setup_params['kernel_initializer'](weights[0].shape)
        weights[1] = model_setup_params['bias_initializer'](weights[1].shape)
        model.layers[-1].set_weights(weights)
        model.optimizer.learning_rate = 0.1

        model.fit(
            train_features_scaled, 
            train_target_scaled, 
            validation_data=(test_features_scaled, test_target_scaled),
            epochs=200, 
            callbacks=callbacks, 
            verbose=1
        )
        model.load_weights(checkpoint_path)
        results_train = model_eval(
            model,
            train_features_scaled,
            train_target,
            target_scaler,
            model_name=model_type,
            fold_idx=idx,
            split_name="SecondFit_Train"
        )
        results_test = model_eval(
            model,
            test_features_scaled,
            test_target,
            target_scaler,
            model_name=model_type,
            fold_idx=idx,
            split_name="SecondFit_Test"
        )
        file_updater(summary_csv_path, results_train, mode='a')
        file_updater(summary_csv_path, results_test, mode='a')   
        del model     
    sys.exit()
