# Locality Sensitive Deep Learner
Companion paper: Yap XH, Raymer M.   Toxicity Prediction using Locality-Sensitive Deep Learner. Comput. Toxicol. *(Under review)*

Locality-sensitive deep learner (LSDL) uses attention mechanism to learn locality of a chemical in a dataset. We hypothesize that toxicity data has a *locally-linear* data structure: local regions with linear feature-target relationship.  

Repository is oraganized as follows:
- Processed toxicity datasets (Tox21, AChEi, CoMPARA, AOT; [1-4]) using CDK and PaDEL descriptors and Morgan2 (ECFP2) fingerprints *(/data/processed)*
- python scripts with helper functions *(/0_code)*
- Synthetic (locally-linear) data generation *(/1_Notebooks/SynthData/SynthData_10dim_generate_dataset.py)*
- Synthetic (locally-linear) experiments *(1_Notebooks/SynthData/SynthData_10dim_clusternoise*)
- python scripts for each dataset (Dataset names: Tox21_CDKPaDEL, AChEi, CoMPARA, AOT_wFP) *(/1_Notebooks/{dataset_name})*. Note that several of these scripts will need to load trained tensorflow models. The trained models are too large to be included in git repository (>0.1-1GB per model depending on dataset size), but will be cloned onto lab computer (bio-cs). Also please check file paths in code files because files have been moved around/re-organized and paths may be broken. 


For ***Tox21_CDKPaDEL*** (*/1_Notebooks/Tox21_CDKPaDEL/*), further preprocessing and training of COSA feature weights was applied in *DataProcessing.ipynb*. The python scripts *Tox21_experiments_{modelName}_singlelabel.py* were used to train models in a few variations: dense (feed-forward neural network), LS (Locality-sensitive model without feature weighting), LSwFW (Locality-sensitive model with COSA feature weighting), LSwFW_ones (Locality-sensitive model with naive feature weighting initialized to ones), xgboost (XGBoost).  


For ***AChEi*** (*/1_Notebooks/AChEi/*), further preprocessing and training of COSA feature weights was applied in *DataProcessing.ipynb*. The models were trained using *210503_AChEi_ExtendedDataset_allFeatures.py*. The combined model was evaluated in *Exploration/ExtendedDatasetAnalyses.ipynb*, under *1.2.2 Plots across all folds* and *1.3 Repeat analysis with humanOnly*.  
An additional model using fingerprints only (*Exploration/AChEi_smiles_humanOnly*) was trained and used to demonstrate the weighting of chemotypes/functional groups on Soman in the thesis and paper.  


For ***AOT_wFP*** (*/1_Notebooks/AOT_wFP*), *AOT_binary.py*, *AOT_multiclass.py*, *AOT_regression.py* were used to generate binary/multi-class/regression deep learners with naming convention similar to Tox21_CDKPaDEL (dense, LS, LSwFW, LSwFW_ones, xgboost).  
__XGBoost models__ are generated from the files *AOT_binary_xgb.py*, *AOT_multiclass_xgb.py*, and *AOT_regression.py*.   
__Combined models__ (Averaged predictions of LSDL with feed-forward neural network) were generated by 1) saving predictions of base classifiers (*Results/AOT_predict/*model*_AOT_*traintest*_predict)*; 2) Running (*GetCombinedModel.py)* to evaluate performance of combined model. This jupyter notebook also outputs predictions of combined model.  
__Stacking models__ are obtained by first tuning the XGBoost classifier with (*Stacking_tuning.py*), then training and evaluating the model(s) with (*Stacking.py*).  


For ***CoMPARA*** (*/1_Notebooks/CoMPARA*), only the binary targets were used. *210427_CoMPARA_binary.py* was used to train deep learners and xgboost. The same workflow as ***AOT_wFP*** was used: 1) Get predictions (saved in *Results/Predicts*) and evaluate combined models (*GetCombinedModel.py*).; 2) Tune Stacking classifier with (*Stacking_tuning.py*) and train Stacking classifier with (*Stacking.py*).  


## Generation of COSA Feature weights

[5-6]

## References
Datasets:
Tox21: [1] R. Huang, M. Xia, Editorial: Tox21 Challenge to Build Predictive Models of Nuclear Receptor and Stress Response Pathways as Mediated by Exposure to Environmental Toxicants and Drugs, Front. Environ. Sci. 5 (2017). https://doi.org/10.3389/fenvs.2017.00003.
AChEi: [2] C.D. Ruark, C.E. Hack, P.J. Robinson, P.E. Anderson, J.M. Gearhart, Quantitative structure-activity relationships for organophosphates binding to acetylcholinesterase, Arch. Toxicol. 87 (2013) 281???289. https://doi.org/10.1007/s00204-012-0934-z.
CoMPARA: [3] K. Mansouri, N. Kleinstreuer, A.M. Abdelaziz, D. Alberga, V.M. Alves, P.L. Andersson, C.H. Andrade, F. Bai, I. Balabin, D. Ballabio, E. Benfenati, B. Bhhatarai, S. Boyer, J. Chen, V. Consonni, S. Farag, D. Fourches, A.T. Garc??a-Sosa, P. Gramatica, F. Grisoni, C.M. Grulke, H. Hong, D. Horvath, X. Hu, R. Huang, N. Jeliazkova, J. Li, X. Li, H. Liu, S. Manganelli, G.F. Mangiatordi, U. Maran, G. Marcou, T. Martin, E. Muratov, D.T. Nguyen, O. Nicolotti, N.G. Nikolov, U. Norinder, E. Papa, M. Petitjean, G. Piir, P. Pogodin, V. Poroikov, X. Qiao, A.M. Richard, A. Roncaglioni, P. Ruiz, C. Rupakheti, S. Sakkiah, A. Sangion, K.W. Schramm, C. Selvaraj, I. Shah, S. Sild, L. Sun, O. Taboureau, Y. Tang, I. V. Tetko, R. Todeschini, W. Tong, D. Trisciuzzi, A. Tropsha, G. Van Den Driessche, A. Varnek, Z. Wang, E.B. Wedebye, A.J. Williams, H. Xie, A. V. Zakharov, Z. Zheng, R.S. Judson, Compara: Collaborative modeling project for androgen receptor activity, Environ. Health Perspect. 128 (2020) 1???17. https://doi.org/10.1289/EHP5580.
AOT: [4] N.C. Kleinstreuer, A.L. Karmaus, K. Mansouri, D.G. Allen, J.M. Fitzpatrick, G. Patlewicz, Predictive models for acute oral systemic toxicity: A workshop to bridge the gap from research to regulation, Comput. Toxicol. 8 (2018) 21???24. https://doi.org/10.1016/j.comtox.2018.08.002.

COSA: [5] J.H. Friedman, J.J. Meulman, Clustering objects on subsets of attributes, J. R. Stat. Soc. Ser. B Stat. Methodol. 66 (2004) 815???839. https://doi.org/10.1111/j.1467-9868.2004.02059.x.
COSA: [6] M.M. Kampert, J.J. Meulman, J.H. Friedman, rCOSA: A Software Package for Clustering Objects on Subsets of Attributes, J. Classif. 34 (2017) 514???547. https://doi.org/10.1007/s00357-017.