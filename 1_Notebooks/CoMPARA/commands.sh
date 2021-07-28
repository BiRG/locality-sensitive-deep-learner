#!/bin/bash

nohup python ./210427_CoMPARA_binary.py --model_type "dense" --resample False > dense_binary.out
nohup python ./210427_CoMPARA_binary.py --model_type "LS" --resample False > LS_binary.out
nohup python ./210427_CoMPARA_binary.py --model_type "LSwFW" --resample False > LSwFW_binary.out
nohup python ./210427_CoMPARA_binary.py --model_type "LSwFW_ones" --resample False> LSwFW_ones_binary.out
