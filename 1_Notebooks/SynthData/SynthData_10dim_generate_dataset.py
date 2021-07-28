#!/usr/bin/env python3
# File name: SynthData_10dim_experiments.py
# Compare dense and locality-sensitive models with increasing proportion of noise features


import os, sys
import numpy as np
import copy
import pandas as pd

code_folder = os.path.join("..", "..", "0_code")
sys.path.append(code_folder)

from data_generation import Synthetic
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

import dill as pickle
from algorithms import COSA

def add_cluster_noise(n_noise_dim, X, data_gen, n_clusters, random_seed=123):
	#Add cluster-specific noise (Should this step go before binary classification or after?)
	## May no longer be a smooth manifold :(

	all_noise_features=[]
	means = [np.mean(X[:,i]) for i in range(X.shape[1])]
	stds = [np.std(X[:,i]) for i in range(X.shape[1])]
	X_noise = copy.deepcopy(X)
	np.random.seed(random_seed)
	for cl in range(n_clusters):
	    cl_ind=np.where(data_gen.cluster_labels==cl)[0]
	    noise_features=np.sort(np.random.choice(range(n_dim), n_noise_dim, replace=False))
	    all_noise_features.append(noise_features)
	    for j in noise_features:
	        X_noise[cl_ind, j]=np.random.normal(means[j], stds[j], size=len(cl_ind))
	# X_noise=X+X_noise
	return X_noise, all_noise_features

def _get_traintest_split(cluster_labels, prop=0.1):
    unique_labels=np.unique(cluster_labels)
    n_test=int(prop*len(cluster_labels))
    n_test_list=[int(n_test/len(unique_labels)) for i in range(len(unique_labels))]
    for i in np.random.choice(unique_labels, replace=False, size=n_test%len(unique_labels)):
        n_test_list[i]+=1
    test_inds=np.concatenate([np.random.choice(np.where(cluster_labels==unique_labels[i])[0],
                                replace=False, 
                                size=n_test_list[i]
                               ) for i in range(len(unique_labels))
                             ])
    test_inds=np.sort(test_inds)
    train_inds=np.setdiff1d(range(len(cluster_labels)), test_inds)
    return train_inds, test_inds	

def get_split(X_noise, data_gen, Fweights=None, random_seed = 123):
	np.random.seed(random_seed)
	X_scaled=preprocessing.scale(X_noise)
	train_inds, test_inds=_get_traintest_split(data_gen.cluster_labels)
	X_train=X_scaled[train_inds]
	X_test=X_scaled[test_inds]
	y_train=y[train_inds]
	y_test=y[test_inds]
	cluster_labels_train=data_gen.cluster_labels[train_inds]
	cluster_labels_test=data_gen.cluster_labels[test_inds]
	if Fweights is not None:
		Fweights_train = Fweights[train_inds]
		Fweights_test = Fweights[test_inds]
		return X_train, X_test, y_train, y_test, cluster_labels_train, cluster_labels_test, Fweights_train, Fweights_test
	return X_train, X_test, y_train, y_test, cluster_labels_train, cluster_labels_test

n_points=1000
n_dim=10
n_clusters=25
data_gen=Synthetic.LocallyLinearManifold(n_points=n_points,
                               n_dim=n_dim,
                               n_clusters=n_clusters,
                               n_knn=10,
                               n_shape_manifold=100,
                               d_shape_manifold=0.25,
                               u_gen='random',
                               clf=LogisticRegression(penalty='l2', 
                                                       solver='lbfgs', 
                                                       C=10, 
                                                       class_weight='balanced'
                                                      ),
                               prior=0.1,
                               preprocessing=preprocessing.MinMaxScaler(),
                               random_seed=123,
                               verbose=True
                              )


if __name__ == "__main__":

	X,y=data_gen.fit()
	n_noise_dim_list = range(9)	
	for n_noise_dim in n_noise_dim_list:
		#Add cluster-based noise
		X_noise, all_noise_features = add_cluster_noise(n_noise_dim, X, data_gen, n_clusters, random_seed=123)
		
		#Get Fweights
		cosa_mdl=COSA.NNCosa(
			Fweight_init="uniform", 
			lam=0.2, 
			n_iter=100, 
			distance_measure="inv_exp_dist", 
			calc_D_ijk=False, 
			threads=8
			)
		cosa_mdl.fit(X_noise)
		Fweights=cosa_mdl.Fweight

		#Setup dataset
		X_train, X_test, y_train, y_test, cluster_labels_train, cluster_labels_test, Fweights_train, Fweights_test = get_split(X_noise, data_gen, Fweights=Fweights, random_seed=123)


		#Save datasets
		X_train_df=pd.DataFrame(X_train, columns=[f"feat_{i:0>3d}" for i in range(X_train.shape[1])])
		X_train_df['Type']='Training'
		X_train_df['cluster_labels']=cluster_labels_train
		X_train_df['Class']=y_train

		X_test_df=pd.DataFrame(X_test, columns=[f"feat_{i:0>3d}" for i in range(X_train.shape[1])])
		X_test_df['Type']='Testing'
		X_test_df['cluster_labels']=cluster_labels_test
		X_test_df['Class']=y_test

		df=pd.concat([X_train_df, X_test_df])
		cols=df.columns[np.concatenate([[10,11,12],range(10)])]
		df=df[cols]

		SynthDataFolder="SynthData_10dim_clusternoise_unbalanced"
		to_save = {
			'df':df, 
			'Fweights_train': Fweights_train,
			'Fweights_test': Fweights_test,
			'all_noise_features': all_noise_features
		}
		with open(os.path.join(SynthDataFolder, 
			f"SynthData_10dim_clusternoise_{str(n_noise_dim)}noisedim_unbalanced.ob"), 'wb') as f:
			pickle.dump(to_save, f)
	sys.exit()