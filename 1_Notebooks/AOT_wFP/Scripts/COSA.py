from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform
from itertools import product
# from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
from joblib import parallel_backend, Parallel, delayed
from scipy.stats import beta
import copy
import numpy as np
import gc
from numba import jit
import dill as pickle

from itertools import product
class PerFeatureDistMat(object):
    def __init__(self, X, X_ref=None):
        if X_ref:
            self.X_ref=X_ref #If given, return Per feature dist mat relative to X_ref
            self.n_ref_points=X_ref.shape[0]
            self.D_ijk=self.get_D_ijk_wref(X, X_ref)
        else:    
            self.D_ijk=self.get_D_ijk(X)
        self.n_feat=X.shape[1]
        self.n_points=X.shape[0]

    def get_D_ijk_wref(self, X, X_ref):
        #D_ijk is a matrix that is n_feat x (n_points x n_ref_points distance matrix)
        D_ijk=np.array([cdist(np.vstack(X[:,i], X_ref[:,i])) for i in range(X.shape[1])])
        return D_ijk
    
    def get_D_ijk(self, X):
        #D_ijk is a matrix that is n_feat x (n_points x n_points distance matrix in reduced form)
        D_ijk=np.array([pdist(np.vstack(X[:,i])) for i in range(X.shape[1])])
        return D_ijk        
    
    def __getitem__(self, feat_row_col):
        feat, row, col=feat_row_col[0], feat_row_col[1], feat_row_col[2]        
        if type(feat)==slice:
            feat=range(*feat.indices(self.n_feat))
        if type(row)==slice:
            row=list(range(*row.indices(self.n_points)))
        elif isinstance(row, (np.integer, int)):
            row=[row]
        if type(col)==slice:
            col=range(*col.indices(self.n_points))
        elif isinstance(col, (np.integer, int)):
            col=[col]

        out=self.__getdist(feat, row, col)
        if len(row)==len(col)==1:
            out=out.flatten()
        return out
    
    def __getdist(self, feat, row, col):
        if self.X_ref is None:
            condensed_ind=list(map(lambda x: self.__get_condensed_idx(x, self.n_points), 
                                   product(row,col)))
            zero_ind=np.where(np.array(condensed_ind)<0)[0]
            out=[self.D_ijk[x][condensed_ind] for x in feat]
            for x in feat:
                out[x][zero_ind]=0.
        else:
            inds=list(product(row, col))
            out = [self.D_ijk[x][inds] for x in feat]
        return np.array(out)
    
    @staticmethod
    def __get_condensed_idx(x,n):
        i,j=x
        if i<j:
            condensed_idx = i*n + j - i*(i+1)/2 - i - 1
        elif i>j:
            condensed_idx = j*n + i - j*(j+1)/2 - j - 1
        elif i==j:
            condensed_idx = -1
        return int(condensed_idx)

class NNCosa(object):
    def __init__(self,
                 Fweight_init="random",
                 n_iter=30,
                 max_inner_iter=10,
                 k=10,
                 lam=0.2,
                 batch_size=0.1, #If batch_size<1, then proportion of n_points. If int, then n_batch.
                 sampling='random',
                 distance_measure="beta_distance",
                 beta_a=0.2,
                 beta_b=9,
                 random_seed=123,
                 calc_D_ijk=True, #Set as False if MemoryError (n_feat x n_points x n_points)
                 threads=8, 
                ):
        self.Fweight_init=Fweight_init
        self.n_iter=n_iter
        self.max_inner_iter=max_inner_iter
        self.k=k #Number of nearest neighbors used for weight update
        self.lam=lam #Regularizer for calculating new weights
        self.batch_size=batch_size
        self.sampling=sampling
        self.distance_measure=distance_measure #beta_distance or inv_exp_dist
        self.beta_a=beta_a #Parameter `a` for beta distribution, used for approximating learning rate
        self.beta_b=beta_b #Parameter `b` for beta distribution, used for approximating learning rate
        self.calc_D_ijk=calc_D_ijk
        self.random_seed=random_seed
        self.threads=threads

    def construct(self, n_points, n_feat, Fweight_init="random", OOS=False):
        #Initialize n_batch
        if self.batch_size<1:
            n_batch=int(self.n_points*self.batch_size)
        else:
            n_batch=int(np.ceil(n_points/self.batch_size))
        if OOS:
            self.n_batch_OOS=int(np.ceil(n_points/self.batch_size))
        else:
            self.n_batch=n_batch
        #Initialize Fweight
        if Fweight_init=="random":
            Fweight=np.random.random(size=(n_points, n_feat))
        elif Fweight_init=="uniform":
            Fweight=np.ones(shape=(n_points,n_feat), dtype=np.float64)
        Fweight=normalize(Fweight, norm='l1', axis=1)*n_feat #L1-norm Instance-based weights
        return Fweight
    
    def fit(self, X, interim_save_file=None):
        X,self.s_k=self._check_X(X)
        self.n_points, self.n_feat = X.shape
        n_points=X.shape[0]
        n_feat=X.shape[1]        
        self.Fweight=self.construct(n_points, n_feat, Fweight_init=self.Fweight_init)
        if self.calc_D_ijk:
            self.D_ijk=PerFeatureDistMat(X)
            #self.D_ijk=self.get_D_ijk(X)
        else:
            self.X=X
            
        np.random.seed(self.random_seed)
        batches=self.rand_batch_inds(n_points, self.n_batch)
        new_Fweight=np.empty_like(self.Fweight)
        for it in range(1, self.n_iter+1):
            if self.distance_measure=="beta_distance":
                beta_it=self.get_beta_it(it/self.n_iter, a=self.beta_a, b=self.beta_b)
            elif self.distance_measure=="inv_exp_dist":
                beta_it=it*self.lam*0.1+ self.lam #if using eta in inv_exp_dist instead
            print(f"Starting on outer iteration {it}; beta/eta:{beta_it:0.3f}")                
            Wchange=1
            inner_it=0
            while Wchange>0.55 and inner_it<self.max_inner_iter:
                with Pool(self.threads) as p:
                    func=lambda x: self._fit_step(*x)
                    ret=p.map(func, list(product(batches, [beta_it])))
#                 with parallel_backend('threading', n_jobs=self.threads):
#                     ret=Parallel()(delayed(self._fit_step)(*arg) for arg in list(product(batches, [beta_it])))
                batches_Fweight=list(map(lambda x: x[0], ret))
#                 crits=list(map(lambda x: x[1], ret))
                #Update new weights
                for idx, batch_ind in enumerate(batches):
                    new_Fweight[batch_ind,:]=copy.deepcopy(batches_Fweight[idx])
                eps=np.finfo(np.float64).eps
#                 crit=np.sum(crits)+np.sum(list(map(lambda x: 
#                     self.lam*np.sum(new_Fweight[x]/self.k*np.log(eps+new_Fweight[x]/self.k)), 
#                                                   range(self.n_points))))
                Wchange=np.average(np.abs(self.Fweight-new_Fweight))
                self.Fweight=copy.deepcopy(new_Fweight)
                print(f"Wchange:{Wchange:.3f}, Crit:Not calculated")
#                 print(f"Wchange:{Wchange:.3f}, Crit:{crit:.3f}")            
                inner_it=inner_it+1
            if interim_save_file is not None:
                r = {'Fweight': self.Fweight}
                with open(interim_save_file, 'wb') as f:
                    pickle.dump(r,f)
            print(f"Inner loop converged in (or maxed out) at {inner_it} steps")
                
                #self.Fweight[batch_ind,:]=copy.deepcopy(batches_Fweight[idx]) 
                
    def get_D_ijk(self, X):
        #D_ijk is a matrix that is n_feat x (n_points x n_points distance matrix in reduced form)
        D_ijk=np.array([squareform(pdist(np.vstack(X[:,i]))) for i in range(X.shape[1])])
        return D_ijk     
    
    def output_Dmat(self, OOS=False):    
        if OOS=="all":
            n_all=self.n_points+self.n_OOS_points
            batches=np.array_split(range(n_all), n_all)
        elif OOS:
            batches=np.array_split(range(self.n_OOS_points), self.n_batch_OOS)
        else:
            batches=np.array_split(range(self.n_points), self.n_batch)
        if self.distance_measure=="beta_distance":
            beta_it=1. #beta set at 1 for output distance matrix
        elif self.distance_measure=="inv_exp_dist":
            beta_it=self.n_iter*0.02+self.lam
#         with Pool(self.threads) as p:
#             Dmat_batches=p.starmap(self._get_Dmat, list(product(batches, [beta_it], [OOS]))) 
        with parallel_backend('threading', n_jobs=self.threads):
            Dmat_batches=Parallel()(delayed(self._get_Dmat)(*arg) for arg in list(product(batches, [beta_it], [OOS])))
        Dmat=np.empty_like(np.vstack(Dmat_batches))
        for i in range(len(batches)):
            Dmat[batches[i]] =Dmat_batches[i]
        #Dmat=np.reshape(Dmat, (self.n_points, self.n_feat))
        return Dmat
    
    def output_Fweight(self, OOS=False):
        if OOS=="all":
            return np.vstack([self.Fweight, self.Fweight_OOS])
        elif OOS:
            return self.Fweight_OOS
        else:
            return self.Fweight

    def fit_OOS(self, X_OOS, interim_save_file=None):
        X_OOS=X_OOS[:,self.valid_feat]/self.s_k
        self.n_OOS_points=X_OOS.shape[0]
        n_feat=X_OOS.shape[1]
        assert n_feat==self.X.shape[1], "Number of features in out of sample datapoints must match original input data."
        
        self.Fweight_OOS=self.construct(self.n_OOS_points, n_feat, Fweight_init=self.Fweight_init,
                                       OOS=True)
        if self.calc_D_ijk:
            self.D_ijk=PerFeatureDistMat(X_OOS, self.X)
        else:
            self.X_OOS=X_OOS
        np.random.seed(self.random_seed)
        batches=self.rand_batch_inds(self.n_OOS_points, self.n_batch)
        new_Fweight_OOS=np.empty_like(self.Fweight_OOS)
        
        for it in range(1, self.n_iter+1):
            if self.distance_measure=="beta_distance":
                beta_it=self.get_beta_it(it/self.n_iter, a=self.beta_a, b=self.beta_b)
            elif self.distance_measure=="inv_exp_dist":
                beta_it=it*0.02+0.2 #if using eta in inv_exp_dist instead            
            print(f"Starting on outer iteration {it}; beta/eta:{beta_it:0.3f}")                
            Wchange=1
            inner_it=0
            while Wchange>0.55 and inner_it<self.max_inner_iter:
#                 with Pool(self.threads) as p:
#                     ret=p.starmap(
#                         self._fit_step, 
#                         list(product(batches, [beta_it], [True])))
                with parallel_backend('threading', n_jobs=self.threads):
                    ret=Parallel()(delayed(self._fit_step)(*arg) 
                                   for arg in list(product(batches, [beta_it], [True])))
                batches_Fweight_OOS=list(map(lambda x: x[0], ret))
#                 crits=list(map(lambda x: x[1], ret))  
#                 #Calculate crit 
#                 eps=np.finfo(np.float64).eps                
#                 crit=np.sum(crits)+np.sum(list(map(lambda x: 
#                     self.lam*np.sum(new_Fweight_OOS[x]/self.k*np.log(eps+new_Fweight_OOS[x]/self.k)), 
#                                                   range(self.n_OOS_points))))
                #Update new weights
                for idx, batch_ind in enumerate(batches):
                    new_Fweight_OOS[batch_ind,:]=copy.deepcopy(batches_Fweight_OOS[idx])

                Wchange=np.sum(np.abs(self.Fweight_OOS-new_Fweight_OOS))
                self.Fweight_OOS=copy.deepcopy(new_Fweight_OOS)
#                 print(f"Wchange:{Wchange:.3f}, Crit:{crit:.3f}")
                print(f"Wchange:{Wchange:.3f}, Crit: Not calculated")
                inner_it=inner_it+1
                #self.Fweight[batch_ind,:]=copy.deepcopy(batches_Fweight[idx])     
            print(f"Inner loop converged in (or maxed out) at {inner_it} steps")
            if interim_save_file is not None:
                r = {'Fweights_OOS': self.Fweights_OOS}
                with open(interim_save_file, 'w') as f:
                    pickle.dump(r, f)
            
    def _fit_step(self, batch_ind, beta_it, OOS=False):
        #1. Get distances 'activation' Dmat (n_batch x n_points)
        Dmat=self._get_Dmat(batch_ind, beta_it, OOS=OOS)
        
        #2. Get knn (n_batch x k)
        knnMat=np.argsort(Dmat, axis=1)
        if OOS:
            knnMat=knnMat[:,:self.k]
        else:
            #Remove self-predict
            knnMat=knnMat[:,:(self.k+1)]
            knnMat=list(map(lambda x: np.delete(knnMat[x,:], 
                                        np.where(knnMat[x,:]==batch_ind[x])[0]) 
                             if batch_ind[x] in knnMat[x,:] 
                             else knnMat[x,:-1], 
                             range(knnMat.shape[0])))
            knnMat=np.array(list(map(lambda x: np.sort(x), knnMat)))
        #3. Get new weights (Consider changing to weight updates)   
        if OOS:
            n_OOS=knnMat.shape[0]
            #args=[self.Fweight[knnMat[i],:] for i in range(n_OOS)]
            #Wmat=np.array(list(map(lambda arg: self.calc_w_OOS(arg), args)))
            args=[(np.ones(self.k), np.abs(self.X_OOS[batch_ind[i]]-self.X[j_list]),self.lam)
                     for i, j_list in enumerate(knnMat)]
            Wmat=np.array(list(map(lambda arg: calc_cosa_w(*arg), args)))
            
        else:
            if self.calc_D_ijk:
                args=[(np.ones(self.k), np.transpose(self.D_ijk[:,batch_ind[i],j_list]), self.lam) 
                          for i, j_list in enumerate(knnMat)]
            else:
                args=[(np.ones(self.k), np.abs(self.X[batch_ind[i]]-self.X[j_list]), self.lam)
                         for i, j_list in enumerate(knnMat)]
            Wmat=np.array(list(map(lambda arg: calc_cosa_w(*arg), args)))
            
        #4. Get crit
#         crit=np.sum(list(map(lambda i: np.sum(Dmat[i,knnMat[i]]), range(len(batch_ind)))))/self.k
        crit="not calculated"
        del Dmat, knnMat
        return Wmat, crit
    

    def _check_X(self, X):
        n,p=X.shape
        #Drops features that has dispersion=0
        args=[(X[:,i]) for i in range(p)]
#         with Pool(self.threads) as p1:
#             s_k=p1.map(calc_sk2, args)
        with parallel_backend('threading', n_jobs=self.threads):
            s_k=Parallel()(delayed(calc_sk2)(arg) for arg in args)
        self.valid_feat=np.where(~np.isclose(s_k,0))[0]
        s_k=np.array(s_k)[self.valid_feat]
        return X[:, self.valid_feat]/s_k, s_k    
    
    def _get_Dmat(self, batch_ind, beta_it, OOS=False):
        if OOS=="all":
            Fweights_all=np.vstack([self.Fweight, self.Fweight_OOS])
            X_all=np.vstack([self.X, self.X_OOS])
            n_all=self.n_points+self.n_OOS_points
            args=[(Fweights_all[i,:], Fweights_all[j,:], np.abs(X_all[i]-X_all[j]), beta_it)
                     for (i,j) in list(product(batch_ind, range(n_all)))]
        elif self.calc_D_ijk:
            if OOS:
                args=[(self.Fweight_OOS[i,:], self.Fweight[j,:], self.D_ijk[:,i,j], beta_it)
                      for (i,j) in list(product(batch_ind, range(self.n_points)))]
            else:
                args=[(self.Fweight[i,:], self.Fweight[j,:], self.D_ijk[:,i,j], beta_it) 
                      for (i,j) in list(product(batch_ind, range(self.n_points)))]                
        else:
            if OOS:
                args=[(self.Fweight_OOS[i,:], self.Fweight[j,:], np.abs(self.X_OOS[i]-self.X[j]), beta_it)
                     for (i,j) in list(product(batch_ind, range(self.n_points)))]                
            else:
                args=[(self.Fweight[i,:], self.Fweight[j,:], np.abs(self.X[i]-self.X[j]), beta_it)
                     for (i,j) in list(product(batch_ind, range(self.n_points)))]
        if self.distance_measure=="beta_distance":
            Dmat=list(map(beta_distance, args))
        elif self.distance_measure=="inv_exp_dist":
            Dmat=list(map(lambda arg: inv_exp_dist(*arg), args))
        
        if OOS=="all":
            Dmat=np.reshape(Dmat, (len(batch_ind), n_all))
        else:
            Dmat=np.reshape(Dmat, (len(batch_ind), self.n_points))    
        return Dmat        
    
    @staticmethod
    def rand_batch_inds(n_points, n_batch):
        if n_points<n_batch+1:
            return [np.arange(n_points)]
        arr=np.array(range(n_points))
        np.random.shuffle(arr)
        batches=np.array_split(arr, n_batch)
        batches=[np.sort(batches[i]) for i in range(n_batch)]
        return batches

    @staticmethod
    def get_beta_it(eta,a=0.2,b=9):
        return beta.cdf(eta,a,b)/2+0.5  
    
    @staticmethod
    def calc_w_OOS(knnWmat):
        w_OOS=np.average(knnWmat, axis=0)
        return w_OOS
    
def beta_distance(arg):
    w_i, w_j, d_ij, beta_it=arg
    #d_ij is a vector of d_ijk
    w_i=w_i/np.sum(w_i)
    w_j=w_j/np.sum(w_j)
    D_ij=np.max([np.dot(w_i, d_ij),
                 np.dot(w_j, d_ij)
                ])
    return D_ij*beta_it    

# def inv_exp_dist(arg):
#     w_i, w_j, d_ij, eta=arg
#     w_i=w_i/np.sum(w_i)
#     w_j=w_j/np.sum(w_j)    
#     #w_k=np.min(np.vstack([w_i, w_j]), axis=0)

#     D_ij=np.max([-eta*np.log(np.dot(w_i, np.exp(-d_ij/eta))), 
#                 -eta*np.log(np.dot(w_j, np.exp(-d_ij/eta))), 
#                  0
#                 ])
#     #D_ij=-eta*np.log(np.dot(w_k, np.exp(-d_ij/eta)))
#     return D_ij

@jit(nopython=True)
def inv_exp_dist(w_i, w_j, d_ij, eta):
    w_i=w_i/np.sum(w_i)
    w_j=w_j/np.sum(w_j)    
    #w_k=np.min(np.vstack([w_i, w_j]), axis=0)

    D_ij=np.maximum(-eta*np.log(np.dot(w_i, np.exp(-d_ij/eta))), 
                -eta*np.log(np.dot(w_j, np.exp(-d_ij/eta))))
    D_ij=np.maximum(D_ij, 0)
    #D_ij=-eta*np.log(np.dot(w_k, np.exp(-d_ij/eta)))
    return D_ij   

# def calc_cosa_w(arg):
#     W_sdij, d_ij_knn, lam= arg
#     #W_sdij is a weight vector (#knn by 1) for d_ij_knn
#     #d_ij_knn is a #knn by #feat matrix of vectors of d_ijk's, where j \elem knn(i)
#     #lam refers to lambda
#     n_knn=d_ij_knn.shape[0]
#     w_i=np.exp(-(np.dot(W_sdij, d_ij_knn)/(n_knn*lam)))#-1)
#     w_i=(w_i/np.sum(w_i))*len(w_i)
#     return w_i

@jit(nopython=True)
def calc_cosa_w(W_sdij, d_ij_knn, lam):
    #W_sdij is a weight vector (#knn by 1) for d_ij_knn
    #d_ij_knn is a #knn by #feat matrix of vectors of d_ijk's, where j \elem knn(i)
    #lam refers to lambda
    n_knn=d_ij_knn.shape[0]
    w_i=np.exp(-(np.dot(W_sdij, d_ij_knn)/np.multiply(n_knn,lam)))#-1)
    w_i=np.multiply((w_i/np.sum(w_i)),len(w_i))
    return w_i

def calc_sk2(X_k):
    #Faster version of calc_sk
    X_k=np.sort(X_k)
    n=len(X_k)
    res=0
    d=0
    for i in range(n):
        res+=(X_k[i]*i-d)
        d+=X_k[i]
    return res/(n*(n-1)/2)

    