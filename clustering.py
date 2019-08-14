import numpy as np
import eigen_gap as eg
import matplotlib.pyplot as plt
import warnings

import copy
from sklearn.decomposition import PCA
from sklearn import cluster as cl
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.manifold import spectral_embedding
#import scipy

from scipy import sparse
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh,eigs
from sklearn.neighbors import kneighbors_graph

from pathlib import Path
from collections import Counter
#Debugger
import pdb
import sys
import traceback

warnings.filterwarnings("ignore")

class SVD: 

	def __init__(self,count_matrix,scaler='std',with_std=False):
		self.scaler = scaler
		self.with_std = with_std
		self.runSVD(count_matrix)

	def runSVD(self,matrix):
		if self.scaler == "mabs":
			scaler_obj = MaxAbsScaler()
			X = scaler_obj.fit_transform(matrix)
		elif self.scaler == "std":
			scaler_obj = StandardScaler(with_std=self.with_std).fit(matrix)
			X = scaler_obj.fit_transform(matrix)

		self.U, self.s , self.v = np.linalg.svd(X,full_matrices=True)

	def getReduced(self,n_components=0):
		U = self.U
		s = self.s
		if n_components == 0:
			n_components = s.shape[0]
		matrix = -np.dot(U[:,:n_components],np.diag(s[:n_components]))

		return matrix

	def saveReduced(self,n_components=0,file='foo.csv'):
		np.savetxt(file, self.getReduced(n_components), fmt='%f',delimiter=",") 

	def getVarianceComposition(self,n=0):
		s = self.s
		var_comp = [0]
		for i in range(s.shape[0]):
			var_comp.append(var_comp[i]+s[i]*s[i])

		var_comp = [i for i in map(lambda x: x/var_comp[-1],var_comp)]

		if n == 0:
			return var_comp
		if 0 < n <= s.shape[0]:
			return var_comp[n]
		return 0

def runPca(count_matrix,n_components,scaler="std",with_std=True):
	
	X = count_matrix

	if scaler == "mabs":
		scaler_obj = MaxAbsScaler()
		X = scaler_obj.fit_transform(X)
	elif scaler == "std":
		scaler_obj = StandardScaler(with_std=with_std).fit(X)
		X = scaler_obj.fit_transform(X)

	pca = PCA(n_components=n_components)
	X_r = pca.fit(X).transform(X)

	return X_r,pca.explained_variance_ratio_

def spectral(data_matrix,n_clusters,n_neighbors=10):


	X_r = data_matrix
	algorithm = 'spectral'

	spectral = cl.SpectralClustering(n_clusters=n_clusters,
		affinity='nearest_neighbors',n_neighbors=n_neighbors)
	spectral.fit(X_r)
	
	y_pred = spectral.labels_.astype(np.int)
	return y_pred

def kmeans(data_matrix,n_clusters):
	X_r = data_matrix

	kmeans = cl.KMeans(n_clusters=n_clusters, random_state=0).fit(X_r)
	
	y_pred = kmeans.labels_

	return y_pred

def dbscan(data_matrix,eps=0.5,min_samples=5):

	result = cl.DBSCAN(eps=eps, min_samples=min_samples).fit(data_matrix)
	return result.labels_

def runClustering(data_matrix,parameters,algorithm='spectral',
						n_clusters=2,eps=0.5,min_samples=2,path=""):

	if(algorithm == 'spectral'):
		return spectral(data_matrix,n_clusters,
				n_neighbors=parameters['n_neighbors'])
	elif(algorithm == 'kMedoids'):
		return kMedoids(data_matrix,n_clusters)
	elif(algorithm == 'dbscan'):
		return dbscan(data_matrix,eps=eps,min_samples=min_samples)

def runRangeClustering(data_matrix,algorithm,parameters,n_clusters,path=""):
	if algorithm == 'spectral':
		best_clustering = []
		best_sil = -1
		best_n = n_clusters[0]

		for n in n_clusters:
			try:
				clustering = spectral(data_matrix,n,n_neighbors=parameters['n_neighbors'])
				avg_silhouette = silhouette_score(data_matrix,clustering) 
				if best_sil < avg_silhouette:
					best_sil = avg_silhouette
					best_clustering = clustering
					best_n = n
			except:
				traceback.print_exc(file=sys.stderr)
				continue
		#print(best_n,best_sil)
	return best_n,best_clustering

def find_n_clusters(count_matrix):
	n_counter = dict()
	max_gap_connected = dict()
	smallest_neighbors = count_matrix.shape[0]
	svd = SVD(count_matrix,scaler='std',with_std=False)
	var_comp = svd.getVarianceComposition()
	
	idx = 0
	while var_comp[idx] < 0.95:
		idx += 1

	cl_range = [i for i in range(idx,count_matrix.shape[1]+1)]

	for n_components in cl_range:				
		data_matrix = svd.getReduced(n_components)

		for factor in np.arange(0.01,0.91,0.01):
			n_neighbors = int(factor*data_matrix.shape[0])
			n_cluster, connected = eg.calculate_gap(data_matrix,
											n_neighbors=n_neighbors)
			if connected:
				max_gap_connected[n_components] = (n_cluster,n_neighbors)
				if n_neighbors < smallest_neighbors:
					smallest_neighbors = n_neighbors
					n_counter = dict()

				if n_neighbors == smallest_neighbors:
					n_counter[n_cluster] = n_counter.get(n_cluster,0) + 1

				break

	n_clusters = max(n_counter.items(),key= lambda x: x[1])[0]			
	n_cluster_candidates = list(filter(lambda x: x[1][1] == smallest_neighbors and x[1][0] == n_clusters,max_gap_connected.items()))
	i = int((len(n_cluster_candidates)-0.25)/2)
	params = n_cluster_candidates[i]

	#Returns n_components, n_clusters, k_neighbors
	return params[0],params[1][0],params[1][1], svd.getReduced(params[0])