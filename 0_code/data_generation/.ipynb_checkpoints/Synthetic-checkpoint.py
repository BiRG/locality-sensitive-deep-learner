#Data generation
from scipy.spatial.distance import pdist, squareform
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.linear_model import LogisticRegression
import copy


class LocallyLinearManifold(object):
    def __init__(self,
                 n_points=1000,
                 n_dim=10,
                 n_clusters=20,
                 n_knn=10,
                 n_shape_manifold=10,
                 d_shape_manifold=0.1,
                 random_seed=123,
                 verbose=False
                ):
        self.n_points=n_points
        self.n_dim=n_dim
        self.n_clusters=n_clusters
        self.n_knn=n_knn
        self.n_shape_manifold=n_shape_manifold
        self.d_shape_manifold=d_shape_manifold
        self.random_seed=random_seed
        self.verbose=verbose
    
    def fit(self):
        np.random.seed(self.random_seed)
        #Generate sampled 2-D disc
        X=generate_disc(self.n_points)
        #Stretched disc
        X=stretch_disc(X, self.n_dim)
        #Get adjacency matrix
        self.adj_mat=get_knn_edges(X, k=self.n_knn)
        #Shape manifold
        seeds=np.random.randint(np.iinfo(np.int32).max, size=self.n_shape_manifold)
        for i in range(self.n_shape_manifold):
            X=shape_manifold(X, d_shape=self.d_shape_manifold, seed=seeds[i])
        
        #Get classification
        self.cluster_labels, self.valid_clusters=get_clusters(X, self.adj_mat, self.n_clusters)
        y=get_classes_from_clusters(X, self.cluster_labels, self.valid_clusters, verbose=self.verbose)
        return X,y

def get_clusters(X, adj_mat, n_clusters):
    #Get geodesic distances
    G=Graph(adj_mat)
    geo_Dmat=G.get_geodesic_Dmat()
    linked = linkage(squareform(geo_Dmat), 'average')
    
    #Getting labels from clusters of 5 or more members
    #grouped=fcluster(linked, 0.2, criterion='distance')
    grouped=fcluster(linked, n_clusters, criterion='maxclust')
    valid_clusters=np.array(list(filter(lambda x: np.count_nonzero(grouped==x)>4, 
                                        range(np.max(grouped)))))

    cluster_labels=np.zeros_like(grouped)
    ind=np.isin(grouped, valid_clusters)
    cluster_labels[ind]=[np.argwhere(valid_clusters==i)[0][0]+1 for i in grouped[ind]]

    return cluster_labels, valid_clusters

def get_classes_from_clusters(X, cluster_labels, valid_clusters, verbose=False):
    #classifiers=[]
    #u1s=[]
    y=np.zeros_like(cluster_labels)
    for i in valid_clusters:
        X_sub=X[cluster_labels==i]
        u1=np.random.normal(size=X.shape[1])
        u1=u1/np.linalg.norm(u1)
        dot=np.dot(X_sub,u1)
        y_sub=(dot>np.average(dot)).astype(int)
        lr=LogisticRegression(solver='lbfgs', penalty='none')
        lr.fit(X_sub, y_sub)
        if verbose:
            print(lr.score(X_sub, y_sub))
            print(y_sub)
        #classifiers.append(copy.deepcopy(lr))
        #u1s.append(copy.deepcopy(u1))
        y[cluster_labels==i]=y_sub    
    return y

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

def shape_manifold(X, d_shape=0.1, seed=123):
    #Shapes the manifold using projection to a principal vector (t) through the function d_shape*sin((2pi/T)(t+dt))
    ##d_shape: degree of perturbation - set this to a smallish value (<<1)
    ##Let T be [0.5*t_range, 1*t_range), to prevent "too many folds" in manifold
    ##Let dt be [-0.5*t_range, 0.5*t_range)
    d=X.shape[1]
    #Generate principal vector
    np.random.seed(seed)
    u=np.random.normal(size=d)
    u=u/np.linalg.norm(u)
    t=np.dot(X, u)
    #Get sine function
    t_range=np.max(t)-np.min(t)
    T=np.random.uniform(0.5*t_range, t_range)
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

        