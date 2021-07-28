#Data generation
from scipy.spatial.distance import pdist, squareform, cdist
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.linear_model import LogisticRegression
import copy

eps=np.finfo(np.float).eps

class LocallyLinearManifold(object):
    def __init__(self,
                 n_points=1000,
                 n_dim=10,
                 n_clusters=20,
                 n_knn=10, 
                 n_shape_manifold=10,
                 d_shape_manifold=0.1,
                 u_gen='random',
                 T_coef=1, #Set smaller values for sharper changes in the manifold 
                 clf=LogisticRegression(penalty='l2', 
                                        solver='lbfgs', 
                                        C=10, 
                                        class_weight='balanced'
                                       ), #Classifier for assigning classes within a cluster
                 prior=0.5, #Prior probability of the class. Set lower for majority-negative, unbalanced class
                 n_iter=10, #Number of iterations to try when assigning classes within a cluster
                 balance_tol=0.1, #Tolerance of how balanced the classes should be. if default, classes within a cluster have to have prior probabilities between [0.4, 0.6]. 
                 preprocessing=None,
                 random_seed=123,
                 seeds=None, #Seeds for each count of shape_manifold
                 verbose=False
                ):
        self.n_points=n_points
        self.n_dim=n_dim
        self.n_clusters=n_clusters
        self.n_knn=n_knn
        self.n_shape_manifold=n_shape_manifold
        self.d_shape_manifold=d_shape_manifold
        self.u_gen=u_gen
        self.T_coef=T_coef
        self.clf=clf
        self.prior=prior
        self.n_iter=n_iter
        self.balance_tol=balance_tol
        self.preprocessing=preprocessing
        self.random_seed=random_seed
        self.seeds=seeds
        self.verbose=verbose
    
    def fit(self):
        np.random.seed(self.random_seed)
        #Generate sampled 2-D disc
        X=generate_disc(self.n_points)
        #Stretched disc
        X=stretch_disc(X, self.n_dim, seed=self.random_seed)
        #Get adjacency matrix
        self.adj_mat=get_knn_edges(X, k=self.n_knn)
        #Shape manifold
        if self.seeds is None:
            self.seeds=np.random.randint(np.iinfo(np.int32).max, size=self.n_shape_manifold)
        for i in range(self.n_shape_manifold):
            X=shape_manifold(X, d_shape=self.d_shape_manifold, seed=self.seeds[i], T_coef=self.T_coef, u_gen=self.u_gen)
        
        #Preprocessing
        if self.preprocessing is not None:
            X=self.preprocessing.fit_transform(X)
        
        #Get classification
        self.cluster_labels, self.valid_clusters=get_clusters(X, self.adj_mat, self.n_clusters)
        y, self.coefs, self.intercepts=get_classes_from_clusters(X, self.cluster_labels, self.valid_clusters, self.clf, self.prior, n_iter=self.n_iter, balance_tol=self.balance_tol, verbose=self.verbose)
        return X,y

def get_clusters(X, adj_mat, n_clusters):
    #Get geodesic distances
    G=Graph(adj_mat)
    geo_Dmat=G.get_geodesic_Dmat()
    linked = linkage(squareform(geo_Dmat), 'average')
    
    #Getting labels from clusters of 5 or more members
    #grouped=fcluster(linked, 0.2, criterion='distance')
    grouped=fcluster(linked, n_clusters, criterion='maxclust')-1
    valid_clusters=np.array(list(filter(lambda x: np.count_nonzero(grouped==x)>4, 
                                        range(n_clusters))))

    cluster_labels=np.ones_like(grouped)*-1
    #ind=np.isin(grouped, valid_clusters)
    #cluster_labels[ind]=[np.argwhere(valid_clusters==i)[0][0] for i in grouped[ind]]
    for i in valid_clusters:
        ind=np.isin(grouped, [i])
        cluster_labels[ind]=i
    return cluster_labels, valid_clusters

def get_classes_from_clusters(X, cluster_labels, valid_clusters, clf, prior, n_iter=10, balance_tol=0.1, verbose=False):
    #classifiers=[]
    #u1s=[]
    coefs=[]
    intercepts=[]
    y=np.zeros_like(cluster_labels)
    for cluster in valid_clusters:
        X_sub=X[cluster_labels==cluster]
#         u1=np.random.normal(size=X.shape[1])
#         u1=u1/np.linalg.norm(u1)
        #get u1 by connecting mean of X_sub to a randomly chosen farthest point (1 out of n_farthest)
        ##Issue: LR with L2 penalty is frequently unable to learn y_sub
        ##Insufficient solution: re-sample u1 until conditions (balanced classification) are met
        ##HELP: NEed a better way to come up with u1
        used_ind=[]
        unbalanced=True
        it=0
        while unbalanced:
            y_sub, used_ind, _= assign_classes(X_sub, used_ind, prior=prior, n_iter=n_iter)
            
            lr=copy.deepcopy(clf)
            lr.fit(X_sub, y_sub)
            y_sub2=lr.predict(X_sub)
            prop_0= np.float(np.unique(y_sub2, return_counts=True)[1][0])/len(y_sub2) #Prop of instances assigned to 0 (or 1 if 0 is empty)
            if prop_0<prior+balance_tol and prop_0>prior-balance_tol:
                unbalanced=False
            it=it+1
            if it>=n_iter:
                print(f"Class probabilities differ from pre-set parameter greater than `balanced_tol`")
                break
                
        if verbose:
            print(f"After {it} iterations: LR1 score: {lr.score(X_sub, y_sub)}")   
            print(f"{np.unique(y_sub2, return_counts=True)}")
        lr2=copy.deepcopy(clf)
        lr2.fit(X_sub, y_sub2)
        y_sub3=lr2.predict(X_sub)
        n_class_change=np.count_nonzero(y_sub3!=y_sub2)
        if verbose:
            print(f"LR2 score: {lr2.score(X_sub, y_sub3)}")
            print(f"No. of reassignments: {n_class_change} out of {len(y_sub2)}")
            print(f"{np.unique(y_sub3, return_counts=True)}")
        #classifiers.append(copy.deepcopy(lr))
        #u1s.append(copy.deepcopy(u1))
        y[cluster_labels==cluster]=y_sub3   
        coefs.append(lr2.coef_)
        intercepts.append(lr2.intercept_)
    return y, coefs, intercepts

def assign_classes(X_sub, used_ind=[], prior=0.5, n_iter=10):
    ave=np.average(X_sub, axis=0)
    #Randomly select one out of the nearest n_iter points to ave
    d=cdist([ave], X_sub)[0]
    head=np.argsort(d)[:n_iter]
    head=X_sub[np.random.choice(head)]
    #Randomly select a point in the dataset, avoiding re-using indices
    tail=np.random.choice(np.setdiff1d(range(len(X_sub)), used_ind))
    used_ind.append(tail)
    tail=X_sub[tail]+eps
    
    u1=tail-head
    u1=u1/np.linalg.norm(u1)
    
    dot=np.dot(X_sub-head,u1)
    if np.random.choice(range(2)): #Randomly decide to go above or below cut-off value
    	percentile=(1.-prior)*100
    	y_sub=(dot>np.percentile(dot, percentile)).astype(int)
    else:
    	percentile=prior*100
    	y_sub=(dot<np.percentile(dot, percentile)).astype(int)
    
    return y_sub, used_ind, u1

def disc_point(a, theta):
    a=np.sqrt(a)
    return [a*np.cos(theta), a*np.sin(theta)]

def generate_disc(n, seed=123):
    np.random.seed(seed)
    a=np.random.uniform(size=n)
    theta=np.random.uniform(low=-np.pi, high=np.pi, size=n)
    X=disc_point(a, theta)
    return np.transpose(X)

def get_edges(X, r=None, percentile=None):
    #Get edges of points that are <=r distance from each other
    Dmat=pdist(X, metric='euclidean')
    if percentile is None:
        percentile=10
    if r is None:
        r=np.percentile(Dmat, 10)
    adj_mat=Dmat
    adj_mat[Dmat>r]=0.
    adj_mat=squareform(adj_mat)
    return adj_mat

def get_knn_edges(X, k=10):
    #Get edges of points that are within 10 nearest neighbors
    Dmat=pdist(X, metric='euclidean')
    Dmat=squareform(Dmat)
    knn_mat=np.argsort(Dmat, axis=1)
    knn_mat=knn_mat[:,1:k+1] #Drop self
    adj_mat=np.zeros_like(Dmat)
    for i in range(adj_mat.shape[0]):
        adj_mat[i,knn_mat[i,:]]=Dmat[i, knn_mat[i,:]]
    adj_mat=np.maximum(adj_mat, adj_mat.transpose())
    return adj_mat
    
def plot_edges(X, adj_mat):
    #Returns plot values for point-point lines
    pass

def stretch_disc(X, d, seed=123):
    #Stretches points sampled from 2D disc (X) into d-dimensions of 2 random principal vectors
    np.random.seed(seed)
    u1,u2=np.random.random(d), np.random.random(d)
    u1,u2=u1/np.linalg.norm(u1), u2/np.linalg.norm(u2)
    X=np.dot(X, np.vstack([u1,u2]))
    return X

def shape_manifold(X, d_shape=0.1, seed=123, T_coef=1, u_gen='random'):
    #Shapes the manifold using projection to a principal vector (t) through the function d_shape*sin((2pi/T)(t+dt))
    ##d_shape: degree of perturbation - set this to a smallish value (<<1)
    ##Let T be [5*t_range, 10*t_range), to prevent "too many folds" in manifold
    ##Let dt be [-0.5*t_range, 0.5*t_range)
    ##T_coef is a multiplier of T (T_coef *T). Set smaller values to get sharper folds in the manifold
    d=X.shape[1]
    #Generate principal vector
    np.random.seed(seed)
    if u_gen=='random':
        u=np.random.normal(size=d)
    else:
        u=u_gen(d)
        print(u)
    u=u/np.linalg.norm(u)
    t=np.dot(X, u)
    #Get sine function
    t_range=np.max(t)-np.min(t)
    T=T_coef*np.random.uniform(5*t_range, 10*t_range)
    dt=np.random.uniform(-0.5*t_range, 0.5*t_range)
    dX=d_shape*np.dot(np.vstack(np.sin(2*np.pi*(t+dt)/T)),[u])
    X=X+dX
    return X
        
    
    
    
# Get global geodesic distance
class Graph():
    def __init__(self, adj_mat, directed=False, max_iter=10, verbose=False):
        if adj_mat.shape==1:
            adj_mat=squareform(adj_mat)
        self.adj_mat=adj_mat
        self.vertices=len(adj_mat[0])
        self.directed=directed
        self.max_iter=max_iter
        self.verbose=verbose
    
    def get_geodesic_Dmat(self):
        if self.directed:
            raise "Directed graph not implemented"
        adj_mat=self.adj_mat
        sorted_edges=self.get_sorted_edges(adj_mat, directed=self.directed)
        geo_Dmat=np.full(adj_mat.shape, np.inf)
        np.fill_diagonal(geo_Dmat, 0.)
        geo_Dmat=self._get_geo_Dmat(geo_Dmat, 
                                    sorted_edges, 
                                    max_iter=self.max_iter, 
                                    verbose=self.verbose)
        return geo_Dmat
        
    @staticmethod
    def get_sorted_edges(adj_mat, directed=False):
        if not directed:
            adj_mat=np.triu(adj_mat)
        E=np.where(~np.isclose(adj_mat,0.))
        Edges=[[E[0][i], E[1][i], adj_mat[E[0][i], E[1][i]]] for i in range(len(E[0]))]
        sorted_edges=[Edges[i] for i in np.argsort(np.array(Edges)[:,2])]
        return sorted_edges
    
    @staticmethod
    def _get_geo_Dmat(geo_Dmat, sorted_edges, max_iter=10, verbose=False):
        prev_inf_count=np.count_nonzero(np.isinf(geo_Dmat))
        inf_count=prev_inf_count-1
        it=1
        while inf_count>0 or inf_count==prev_inf_count or it>max_iter:
            if verbose:
                print(f"Begin iteration {it}")
            for edge in sorted_edges:
                u,v,edge_length=edge
                #1) Update u-v
                if geo_Dmat[u,v]>edge_length:
                    geo_Dmat[u,v]=edge_length
                if geo_Dmat[v,u]>edge_length:
                    geo_Dmat[u,v]=edge_length
                #2) Get all u-v-k
                vk_list=geo_Dmat[v]
                uk_list=geo_Dmat[u]
                uvk_list=edge_length+vk_list
                vuk_list=edge_length+uk_list
                #3) Update geo_Dmat
                geo_Dmat[u,:]=np.min(np.vstack([uk_list, uvk_list]), axis=0)
                geo_Dmat[v,:]=np.min(np.vstack([vk_list, vuk_list]), axis=0)
                #4) Make geo_Dmat symmetric
                geo_Dmat = np.minimum( geo_Dmat, geo_Dmat.transpose() )
            inf_count=np.count_nonzero(np.isinf(geo_Dmat))
            it=it+1
        return geo_Dmat

        