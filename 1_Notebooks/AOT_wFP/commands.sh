#!/bin/bash

# nohup python ./AOT_multiclass.py "dense" > dense_multiclass.out
# nohup python ./AOT_multiclass.py "LS" > LS_multiclass.out
# nohup python ./AOT_multiclass.py "LSwFW"  > LSwFW_multiclass.out
# nohup python ./AOT_multiclass.py "LSwFW_ones" > LSwFW_ones_multiclass.out
# #C:/Windows/System32/shutdown.exe -sg

# python ./Stacking_tuning.py --train_predict_file "dense_AOT_train_predict" --test_predict_file "dense_AOT_test_predict" --model_type "dense"
python ./Stacking.py --train_predict_file "dense_AOT_train_predict" --test_predict_file "dense_AOT_test_predict" --model_type "dense"
python ./Stacking.py --train_predict_file "LS_AOT_train_predict" --test_predict_file "LS_AOT_test_predict" --model_type "LS"
python ./Stacking.py --train_predict_file "LSwFW_AOT_train_predict" --test_predict_file "LSwFW_AOT_test_predict" --model_type "LSwFW"
python ./Stacking.py --train_predict_file "LSwFW_ones_AOT_train_predict" --test_predict_file "LSwFW_ones_AOT_test_predict" --model_type "LSwFW_ones"
python ./Stacking.py --train_predict_file "xgboost_AOT_train_predict" --test_predict_file "xgboost_AOT_test_predict" --model_type "xgboost"
python ./Stacking.py --train_predict_file "dense_AOT_train_predict" --test_predict_file "dense_AOT_test_predict" --model_type "Combined_LS_mean" --train_predict_file2 "LS_AOT_train_predict" --test_predict_file2 "LS_AOT_test_predict" --combine_method "mean"
python ./Stacking.py --train_predict_file "dense_AOT_train_predict" --test_predict_file "dense_AOT_test_predict" --model_type "Combined_LS_stacked" --train_predict_file2 "LS_AOT_train_predict" --test_predict_file2 "LS_AOT_test_predict" --combine_method "stacked"
python ./Stacking.py --train_predict_file "dense_AOT_train_predict" --test_predict_file "dense_AOT_test_predict" --model_type "Combined_LSwFW_mean" --train_predict_file2 "LSwFW_AOT_train_predict" --test_predict_file2 "LSwFW_AOT_test_predict" --combine_method "mean"
python ./Stacking.py --train_predict_file "dense_AOT_train_predict" --test_predict_file "dense_AOT_test_predict" --model_type "Combined_LSwFW_stacked" --train_predict_file2 "LSwFW_AOT_train_predict" --test_predict_file2 "LSwFW_AOT_test_predict" --combine_method "stacked"
python ./Stacking.py --train_predict_file "dense_AOT_train_predict" --test_predict_file "dense_AOT_test_predict" --model_type "Combined_LSwFW_ones_mean" --train_predict_file2 "LSwFW_ones_AOT_train_predict" --test_predict_file2 "LSwFW_ones_AOT_test_predict" --combine_method "mean"
python ./Stacking.py --train_predict_file "dense_AOT_train_predict" --test_predict_file "dense_AOT_test_predict" --model_type "Combined_LSwFW_ones_stacked" --train_predict_file2 "LSwFW_ones_AOT_train_predict" --test_predict_file2 "LSwFW_ones_AOT_test_predict" --combine_method "stacked"
# C:/Windows/System32/shutdown.exe -sg