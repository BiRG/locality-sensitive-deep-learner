#!/usr/bin/env python3
import copy
import itertools
import numpy as np
from skmultilearn.cluster.base import GraphBuilderBase
from scipy.sparse import issparse, csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

class QGraphBuilder(GraphBuilderBase):
    """
    ::Parameters::
    is_weighted
    c: (int/float, >0) Default=1. Scale parameter for tanh function. Larger values result in smaller Q values.     
    use_absolute_values: (bool) Whether to convert Q matrix to np.absolute(Q)
    use_MST: (bool) Whether to return fully-connected edge_map or Maximum spanning tree
    """
    def __init__(self, is_weighted=True, c=1, use_absolute_values=True, use_MST=False, ignore_nan=True):
        super(QGraphBuilder, self).__init__()
        self.is_weighted=is_weighted
        self.c=c        
        self.use_absolute_values=use_absolute_values
        self.use_MST=use_MST
        self.ignore_nan=ignore_nan
        
    def transform(self,y):
        if issparse(y):
            y=y.toarray()
        # if y[0,0]!=int:
        #     y=y.astype(int)
        Q=get_Q(y, self.c, ignore_nan=self.ignore_nan)
        if self.use_absolute_values:
            Q=np.absolute(Q)
        self.Q=copy.deepcopy(Q)
        if self.use_MST:
            self.Q_MST=self.MST(Q)
            self.edge_map=self.edges_to_dict(self.Q_MST)
        else:
            self.edge_map=self.edges_to_dict(Q)
        return self.edge_map

    @staticmethod
    def edges_to_dict(adj):
        i_ind, j_ind=np.where(adj!=0)
        edge_map={}
        for k in range(len(i_ind)):
            i,j=i_ind[k],j_ind[k]
            edge_map[(i,j)]=adj[i,j]
        return edge_map

    @staticmethod
    def MST(adj):
        """Returns maximum spanning tree of the graph of the adjacency matrix.
        The elements of the adjacency matrix contains edge weights (negatives allowed).
        We are retaining heavier weights
        """
        result=minimum_spanning_tree(csr_matrix(np.max(adj)-abs(adj)),overwrite=False).toarray()
        adj[result<=0.]=0
        return adj

def get_valid_ind(y):
    return np.where(np.isfinite(y))[0]

def get_Q(y,c=1, ignore_nan=True):
    # y=y.astype(int) # Don't convert to int as it messes up with np.nan values
    m=y.shape[1]
    Q=np.zeros((m,m))
    for i,j in itertools.combinations(range(m),2):
        y1, y2 = y[:,i], y[:,j]
        if ignore_nan:
            valid_ind = np.intersect1d(get_valid_ind(y1), get_valid_ind(y2))
            y1, y2 = y1[valid_ind], y2[valid_ind]
        a,b=find_b_a(y1, y2)
#         obs=np.zeros((2,2))
#         v,u_obs=np.unique(y[:,[i,j]], axis=0, return_counts=True)
#         for i2 in range(len(v)): #In case there are classes with no observations
#             obs[tuple(v[i2])]=u_obs[i2]
#         a,b, var_a, var_b = estimate_b_a(obs)
        Q[i,j]=np.tanh((np.log(b)-np.log(a))/c)
        Q[j,i]=Q[i,j]
    return Q

def find_b_a(y1,y2):
    pos_ind=np.where(y1==1)[0]
    neg_ind=np.where(y1==0)[0]
    b=y2[pos_ind]
    b=np.sum(b)/len(b)
    a=y2[neg_ind]
    a=np.sum(a)/len(a)
    return a,b

def estimate_b_a(obs):
    '''
    Given 2x2 observations of 2 labels, estimate parameters b and a for a Bernoulli distribution of 2 dependent binary labels
    '''
    #Marginalize Y_1 over Y_2 to obtain p, where Y_1 ~ Bern(p)
    obs=np.reshape(obs,(2,2))
    n=np.sum(obs)
    ((x_00, x_01), (x_10,x_11))=np.divide(obs,n)
    p=np.divide(np.sum(obs[1]),n)
    #Solve for P(Y_2|Y_1=0)=a^n (1-a)^{1-n} and 
        #P(Y_2|Y_1=1)=b^n (1-b)^{1-n}
    a0=1-np.divide(x_00,(1-p))
    a1=np.divide(x_01, (1-p))
    b0=1-np.divide(x_10, p)
    b1=np.divide(x_11,p)
    a=[a0,a1]
    b=[b0,b1]
    return np.mean(a), np.mean(b), np.var(a), np.var(b)