from tensorflow.python.data.ops.dataset_ops import UnaryDataset
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import AUC, Metric 
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.regularizers import l1, l2

class SimilarityBatchingDataset():
    """
    ```
    #Example call    

    attentions=LSwFW_model.layers[1](train_tensor).numpy()
    simbatched=SimilarityBatchingDataset(
        train_data,
        train_targets,
        attentions,
    )
    rearranged_train_data, rearranged_train_targets=simbatched.get_rearranged_tensor()
    ```
    """     
    def __init__(self, 
                 numpy_dataset,
                 numpy_targets,                 
                 attentions,
                 n_batch=8,
                 clusterer_kwargs={"n_clusters":None,
                                   "affinity":"cosine", 
                                   "linkage":"average"
                                  },
                 use_inter_op_parallelism=True,
                 preserve_cardinality=False,
                 use_legacy_function=False,
                 **kwargs
                ):
        self.numpy_dataset=numpy_dataset
        self.numpy_targets=numpy_targets
        self.attentions=attentions
        self.n_batch=n_batch
        self.clusterer_kwargs=clusterer_kwargs
        self._use_inter_op_parallelism=use_inter_op_parallelism
        self._preserve_cardinality=preserve_cardinality
        self._use_legacy_function=use_legacy_function
    
    

    def _partitions(self):
        #Set n_clusters if not initialized
        if self.clusterer_kwargs.get("n_clusters") is None:
            self.clusterer_kwargs["n_clusters"]=20
        
        clusterer=AgglomerativeClustering(**self.clusterer_kwargs)
        clusterer.fit(self.attentions)
        cl_labels=clusterer.labels_
        unique,counts=np.unique(cl_labels, return_counts=True)
        print(f"Fitted {len(unique)} clusters with distribution {np.sort(counts)[::-1]}")
        #Remember to shuffle partitions
        partitions=[np.where(cl_labels==i)[0] for i in np.unique(cl_labels)]
        m=len(partitions)
        rand_order=np.random.choice(range(m), size=m, replace=False)
        partitions=[partitions[i] for i in rand_order]
        
        return partitions

    def get_rearranged_tensor(self):
        arr=np.empty_like(self.numpy_dataset)
        arr_targets=np.empty_like(self.numpy_targets)
        self.partitions=self._partitions()
        ptr=0
        for p in self.partitions:
            arr[ptr:ptr+len(p)]=self.numpy_dataset[p,:]
            arr_targets[ptr:ptr+len(p)]=self.numpy_targets[p,:]
            ptr+=len(p)
        return arr, arr_targets

    



#Updated 6th Jan 2021 (Edited Line 71 to Line 72. Reduce_mean instead of mean, to preserve the required rank)

from tensorflow.keras import backend as K
import tensorflow as tf
epsilon=K.epsilon

def get_weights_dicts(Y):
    weights_dicts=[]
    for j in range(Y.shape[1]):
        weight_zero, weight_one = _get_label_weights(Y[:,j])
        d={'weight_zero':weight_zero,
           'weight_one':weight_one
          }
        weights_dicts.append(d)
    return weights_dicts
def _get_label_weights(y):
    #Get label weights for majority and minority class using the following:
        #major_weight=n/(n_major*2)
        #minor_weight=n*(n_major/n_minor)/(n_major*2)
        #NaN weights are set to zero
    y1=y[~np.isnan(y)]
    n=len(y1)
    n_zero=np.count_nonzero(np.isclose(y1,0))
    n_one=np.count_nonzero(np.isclose(y1,1))
    if n_zero>n_one:
        weight_zero=n/(n_zero*2)
        weight_one=n*(n_zero/n_one)/(n_zero*2)
    else:
        weight_zero=n*(n_one/n_zero)/(n_one*2)
        weight_one=n/(n_one*2)
    return weight_zero, weight_one    

class BinaryCrossEntropyIgnoreNan(tf.keras.losses.Loss):
    def __init__(self, weights_dicts=None, axis=0, **kwargs):
        super(BinaryCrossEntropyIgnoreNan, self).__init__(**kwargs)
        self.weights_dicts = weights_dicts
        self.axis = axis

    def __call__(self, target, output, sample_weight=None):
        # Binary cross entropy that ignores Nan and replaces with mini-batch Nan with 0
        # modified from tf.python.keras.backend.binary_crossentropy

        # NEED TO TEST THIS CODE MORE THOROUGHLY
        target = tf.convert_to_tensor(target)
        output = tf.convert_to_tensor(output)
        if len(target.shape) == 1:
            target = tf.expand_dims(target, 1)
            output = tf.expand_dims(output, 1)
        epsilon_ = tf.constant(epsilon(), dtype=output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon_, 1. - epsilon_)
        
        #Set logged_output as 0. at indices where target is nan
        logged_output1 = tf.where(tf.math.is_nan(target), 0., tf.math.log(output+epsilon_))
        logged_output2 = tf.where(tf.math.is_nan(target), 0., tf.math.log(1-output+epsilon_))
        
        # Compute cross entropy from probabilities
        bce = tf.math.multiply_no_nan(target, logged_output1) + \
                tf.math.multiply_no_nan(1-target, logged_output2)
        
#         bce = target * tf.math.log(output+epsilon_) + \
#             (1-target) * tf.math.log(1-output+epsilon_)

        bce = tf.math.multiply_no_nan(bce, -1.)

        if self.weights_dicts is not None:
            sample_weight = tf.cast(tf.where(target == 0., 1., 0.)*[
                                    self.weights_dicts[i]['weight_zero'] for i in range(len(self.weights_dicts))], dtype=target.dtype)
            sample_weight += tf.cast(tf.where(target == 1., 1., 0.)*[
                                     self.weights_dicts[i]['weight_one'] for i in range(len(self.weights_dicts))], dtype=target.dtype)
            bce = tf.multiply(sample_weight, bce)
#         return tf.keras.backend.mean(bce, axis=self.axis)
        return tf.math.reduce_sum(bce)

    def call(self, target, output, sample_weight=None):
        return self(target, output, sample_weight=None)

    def get_config(self):
        config = {
            'weights_dicts': self.weights_dicts,
            'axis': self.axis
        }
        return config

class AveragedAUCIgnoreNan(Metric):
    """ 
    Creates a tf.keras.metrics.AUC for each label, using sample_weights to "ignore" NaNs 
    """
    def __init__(self,
                 num_labels,
                 average='macro',
                 AUC_kwargs={'num_thresholds': 200,
                             'curve': "ROC",
                             'summation_method': "interpolation",
                             'name': None,
                             'dtype': None,
                             'thresholds': None,
                             'multi_label': False,
                             'label_weights': None,
                             },
                 **kwargs
                 ):
        self.num_labels = num_labels
        self.average = average
        self.AUC_kwargs = AUC_kwargs
        
        self.AUCs = []
        for i in range(num_labels):
            self.AUCs.append(AUC(**AUC_kwargs))
        super(AveragedAUCIgnoreNan, self).__init__(name = "averaged_auc_ignore_nan")

    def update_state(self, Y_true, Y_pred, sample_weight=None):
        error_msg = f"Y_true and Y_pred should have the same shape, not {Y_true.shape} and {Y_pred.shape}."
        assert Y_true.shape[1:] == Y_pred.shape[1:], error_msg
        assert Y_true.shape[1] == self.num_labels, "Y_true should have same number of labels as initialized."

        for i in range(self.num_labels):
            y_true, y_pred = Y_true[:, i], Y_pred[:, i]
            sample_weight = ~tf.math.is_nan(y_true)
            self.AUCs[i].update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        if self.average == 'macro': #Unweighted mean of AUC-ROCs
            results = []
            for i in range(self.num_labels):
                results.append(self.AUCs[i].result())
            return tf.reduce_mean(results, axis=0)
        
        if self.average == 'micro': #Global metric by summing TP, FP, FN
            micro_AUC = AUC(**AUC_kwargs)
            micro_TP, micro_FP, micro_FN, micro_TN = self.get_micro_values()
            micro_AUC.true_positives = micro_TP
            micro_AUC.false_positives = micro_FP
            micro_AUC.false_negatives = micro_FN
            micro_AUC.true_negatives = micro_TN
            return micro_AUC.result()
    
    def get_micro_values(self):
        micro_true_positives = tf.reduce_sum(
            tf.stack([self.AUCs[i].true_positives for i in range(self.num_labels)], ), 
            axis=0
        )        
        micro_false_positives = tf.reduce_sum(
            tf.stack([self.AUCs[i].false_positives for i in range(self.num_labels)], ), 
            axis=0
        )   
        micro_false_negatives = tf.reduce_sum(
            tf.stack([self.AUCs[i].false_negatives for i in range(self.num_labels)], ), 
            axis=0
        )   
        micro_true_negatives = tf.reduce_sum(
            tf.stack([self.AUCs[i].true_negatives for i in range(self.num_labels)], ), 
            axis=0
        )   
        return micro_TP, micro_FP, micro_FN, micro_TN
    
    def reset_states(self):
        for i in range(self.num_labels):
            self.AUCs[i].reset_states()
    
    def get_config(self):
        config = {
            'num_labels': self.num_labels,
            'average': self.average, 
            'AUC_kwargs': self.AUC_kwargs,
        }
        base_config = super(AveragedAUCIgnoreNan, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def __getstate__(self):
        return self.get_config()

    def __setstate__(self, d):
        for slot in d:
            setattr(self, slot, d[slot])
            
    def __reduce__(self):
        # For `pickle` to unpickle object by calling AveragedAUCIgnoreNan(self.num_labels,self.average, self.AUC_kwargs)
        return (AveragedAUCIgnoreNan, (self.num_labels, 
                                       self.average, 
                                       self.AUC_kwargs
                                      ))