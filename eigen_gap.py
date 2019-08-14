# Code adapted from scikit-learn v0.21
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/manifold/spectral_embedding_.py

import warnings

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, lobpcg
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import laplacian as csgraph_laplacian

from sklearn.utils import check_random_state, check_array, check_symmetric
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.extmath import _deterministic_vector_sign_flip

import matplotlib.pyplot as plt
from pathlib import Path

def _graph_connected_component(graph, node_id):
    """Find the largest graph connected components that contains one
    given node
    Parameters
    ----------
    graph : array-like, shape: (n_samples, n_samples)
        adjacency matrix of the graph, non-zero weight means an edge
        between the nodes
    node_id : int
        The index of the query node of the graph
    Returns
    -------
    connected_components_matrix : array-like, shape: (n_samples,)
        An array of bool value indicating the indexes of the nodes
        belonging to the largest connected components of the given query
        node
    """
    n_node = graph.shape[0]
    if sparse.issparse(graph):
        # speed up row-wise access to boolean connection mask
        graph = graph.tocsr()
    connected_nodes = np.zeros(n_node, dtype=np.bool)
    nodes_to_explore = np.zeros(n_node, dtype=np.bool)
    nodes_to_explore[node_id] = True
    for _ in range(n_node):
        last_num_component = connected_nodes.sum()
        np.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)
        if last_num_component >= connected_nodes.sum():
            break
        indices = np.where(nodes_to_explore)[0]
        nodes_to_explore.fill(False)
        for i in indices:
            if sparse.issparse(graph):
                neighbors = graph[i].toarray().ravel()
            else:
                neighbors = graph[i]
            np.logical_or(nodes_to_explore, neighbors, out=nodes_to_explore)
    return connected_nodes


def _graph_is_connected(graph):
    """ Return whether the graph is connected (True) or Not (False)
    Parameters
    ----------
    graph : array-like or sparse matrix, shape: (n_samples, n_samples)
        adjacency matrix of the graph, non-zero weight means an edge
        between the nodes
    Returns
    -------
    is_connected : bool
        True means the graph is fully connected and False means not
    """
    if sparse.isspmatrix(graph):
        # sparse graph, find all the connected components
        n_connected_components, _ = connected_components(graph)
        return n_connected_components == 1
    else:
        # dense graph, find all connected components start from node 0
        return _graph_connected_component(graph, 0).sum() == graph.shape[0]


def _set_diag(laplacian, value, norm_laplacian):
	"""Set the diagonal of the laplacian matrix and convert it to a
	sparse format well suited for eigenvalue decomposition
	Parameters
	----------
	laplacian : array or sparse matrix
		The graph laplacian
	value : float
		The value of the diagonal
	norm_laplacian : bool
		Whether the value of the diagonal should be changed or not
	Returns
	-------
	laplacian : array or sparse matrix
		An array of matrix in a form that is well suited to fast
		eigenvalue decomposition, depending on the band width of the
		matrix.
	"""
	n_nodes = laplacian.shape[0]
	# We need all entries in the diagonal to values
	if not sparse.isspmatrix(laplacian):
		if norm_laplacian:
			laplacian.flat[::n_nodes + 1] = value
	else:
		laplacian = laplacian.tocoo()
		if norm_laplacian:
			diag_idx = (laplacian.row == laplacian.col)
			laplacian.data[diag_idx] = value
		# If the matrix has a small number of diagonals (as in the
		# case of structured matrices coming from images), the
		# dia format might be best suited for matvec products:
		n_diags = np.unique(laplacian.row - laplacian.col).size
		if n_diags <= 7:
			# 3 or less outer diagonals on each side
			laplacian = laplacian.todia()
		else:
			# csr has the fastest matvec and is thus best suited to
			# arpack
			laplacian = laplacian.tocsr()
	return laplacian


def spectral_embedding(adjacency, n_components=8, eigen_solver=None,
					   random_state=None, eigen_tol=0.0,
					   norm_laplacian=True, drop_first=True):
	"""Project the sample on the first eigenvectors of the graph Laplacian.
	The adjacency matrix is used to compute a normalized graph Laplacian
	whose spectrum (especially the eigenvectors associated to the
	smallest eigenvalues) has an interpretation in terms of minimal
	number of cuts necessary to split the graph into comparably sized
	components.
	This embedding can also 'work' even if the ``adjacency`` variable is
	not strictly the adjacency matrix of a graph but more generally
	an affinity or similarity matrix between samples (for instance the
	heat kernel of a euclidean distance matrix or a k-NN matrix).
	However care must taken to always make the affinity matrix symmetric
	so that the eigenvector decomposition works as expected.
	Note : Laplacian Eigenmaps is the actual algorithm implemented here.
	Read more in the :ref:`User Guide <spectral_embedding>`.
	Parameters
	----------
	adjacency : array-like or sparse matrix, shape: (n_samples, n_samples)
		The adjacency matrix of the graph to embed.
	n_components : integer, optional, default 8
		The dimension of the projection subspace.
	eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}, default None
		The eigenvalue decomposition strategy to use. AMG requires pyamg
		to be installed. It can be faster on very large, sparse problems,
		but may also lead to instabilities.
	random_state : int, RandomState instance or None, optional, default: None
		A pseudo random number generator used for the initialization of the
		lobpcg eigenvectors decomposition.  If int, random_state is the seed
		used by the random number generator; If RandomState instance,
		random_state is the random number generator; If None, the random number
		generator is the RandomState instance used by `np.random`. Used when
		``solver`` == 'amg'.
	eigen_tol : float, optional, default=0.0
		Stopping criterion for eigendecomposition of the Laplacian matrix
		when using arpack eigen_solver.
	norm_laplacian : bool, optional, default=True
		If True, then compute normalized Laplacian.
	drop_first : bool, optional, default=True
		Whether to drop the first eigenvector. For spectral embedding, this
		should be True as the first eigenvector should be constant vector for
		connected graph, but for spectral clustering, this should be kept as
		False to retain the first eigenvector.
	Returns
	-------
	embedding : array, shape=(n_samples, n_components)
		The reduced samples.
	Notes
	-----
	Spectral Embedding (Laplacian Eigenmaps) is most useful when the graph
	has one connected component. If there graph has many components, the first
	few eigenvectors will simply uncover the connected components of the graph.
	References
	----------
	* https://en.wikipedia.org/wiki/LOBPCG
	* Toward the Optimal Preconditioned Eigensolver: Locally Optimal
	  Block Preconditioned Conjugate Gradient Method
	  Andrew V. Knyazev
	  https://doi.org/10.1137%2FS1064827500366124
	"""
	adjacency = check_symmetric(adjacency)
	#print(adjacency)
	try:
		from pyamg import smoothed_aggregation_solver
	except ImportError:
		if eigen_solver == "amg":
			raise ValueError("The eigen_solver was set to 'amg', but pyamg is "
							 "not available.")

	if eigen_solver is None:
		eigen_solver = 'arpack'
	elif eigen_solver not in ('arpack', 'lobpcg', 'amg'):
		raise ValueError("Unknown value for eigen_solver: '%s'."
						 "Should be 'amg', 'arpack', or 'lobpcg'"
						 % eigen_solver)

	random_state = check_random_state(random_state)

	n_nodes = adjacency.shape[0]
	# Whether to drop the first eigenvector
	if drop_first:
		n_components = n_components + 1

	connected = True
	if not _graph_is_connected(adjacency):
		# warnings.warn("Graph is not fully connected, spectral embedding"
		# 			  " may not work as expected.")
		connected = False

	laplacian, dd = csgraph_laplacian(adjacency, normed=norm_laplacian,
									  return_diag=True)
	if (eigen_solver == 'arpack' or eigen_solver != 'lobpcg' and
	   (not sparse.isspmatrix(laplacian) or n_nodes < 5 * n_components)):
		# lobpcg used with eigen_solver='amg' has bugs for low number of nodes
		# for details see the source code in scipy:
		# https://github.com/scipy/scipy/blob/v0.11.0/scipy/sparse/linalg/eigen
		# /lobpcg/lobpcg.py#L237
		# or matlab:
		# https://www.mathworks.com/matlabcentral/fileexchange/48-lobpcg-m
		laplacian = _set_diag(laplacian, 1, norm_laplacian)

		# Here we'll use shift-invert mode for fast eigenvalues
		# (see https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
		#  for a short explanation of what this means)
		# Because the normalized Laplacian has eigenvalues between 0 and 2,
		# I - L has eigenvalues between -1 and 1.  ARPACK is most efficient
		# when finding eigenvalues of largest magnitude (keyword which='LM')
		# and when these eigenvalues are very large compared to the rest.
		# For very large, very sparse graphs, I - L can have many, many
		# eigenvalues very near 1.0.  This leads to slow convergence.  So
		# instead, we'll use ARPACK's shift-invert mode, asking for the
		# eigenvalues near 1.0.  This effectively spreads-out the spectrum
		# near 1.0 and leads to much faster convergence: potentially an
		# orders-of-magnitude speedup over simply using keyword which='LA'
		# in standard mode.
		try:
			# We are computing the opposite of the laplacian inplace so as
			# to spare a memory allocation of a possibly very large array
			laplacian *= -1
			v0 = random_state.uniform(-1, 1, laplacian.shape[0])
			lambdas, diffusion_map = eigsh(laplacian, k=n_components,
										   sigma=1.0, which='LM',
										   tol=eigen_tol, v0=v0)
			embedding = diffusion_map.T[n_components::-1]
			if norm_laplacian:
				embedding = embedding / dd
		except RuntimeError:
			# When submatrices are exactly singular, an LU decomposition
			# in arpack fails. We fallback to lobpcg
			eigen_solver = "lobpcg"
			# Revert the laplacian to its opposite to have lobpcg work
			laplacian *= -1

	if eigen_solver == 'amg':
		# Use AMG to get a preconditioner and speed up the eigenvalue
		# problem.
		if not sparse.issparse(laplacian):
			warnings.warn("AMG works better for sparse matrices")
		# lobpcg needs double precision floats
		laplacian = check_array(laplacian, dtype=np.float64,
								accept_sparse=True)
		laplacian = _set_diag(laplacian, 1, norm_laplacian)
		ml = smoothed_aggregation_solver(check_array(laplacian, 'csr'))
		M = ml.aspreconditioner()
		X = random_state.rand(laplacian.shape[0], n_components + 1)
		X[:, 0] = dd.ravel()
		lambdas, diffusion_map = lobpcg(laplacian, X, M=M, tol=1.e-12,
										largest=False)
		embedding = diffusion_map.T
		if norm_laplacian:
			embedding = embedding / dd
		if embedding.shape[0] == 1:
			raise ValueError

	elif eigen_solver == "lobpcg":
		# lobpcg needs double precision floats
		laplacian = check_array(laplacian, dtype=np.float64,
								accept_sparse=True)
		if n_nodes < 5 * n_components + 1:
			# see note above under arpack why lobpcg has problems with small
			# number of nodes
			# lobpcg will fallback to eigh, so we short circuit it
			if sparse.isspmatrix(laplacian):
				laplacian = laplacian.toarray()
			lambdas, diffusion_map = eigh(laplacian)
			embedding = diffusion_map.T[:n_components]
			if norm_laplacian:
				embedding = embedding / dd
		else:
			laplacian = _set_diag(laplacian, 1, norm_laplacian)
			# We increase the number of eigenvectors requested, as lobpcg
			# doesn't behave well in low dimension
			X = random_state.rand(laplacian.shape[0], n_components + 1)
			X[:, 0] = dd.ravel()
			lambdas, diffusion_map = lobpcg(laplacian, X, tol=1e-15,
											largest=False, maxiter=2000)
			embedding = diffusion_map.T[:n_components]
			if norm_laplacian:
				embedding = embedding / dd
			if embedding.shape[0] == 1:
				raise ValueError

	embedding = _deterministic_vector_sign_flip(embedding)
	if drop_first:
		return embedding[1:n_components].T,lambdas,connected
	else:
		return embedding[:n_components].T,lambdas,connected

def plot_gap(eigenvalues,filename='eigen.png',nb_clusters=None):
	"""
	:param A: Affinity matrix
	:param plot: plots the sorted eigen values for visual inspection
	:return A tuple containing:
	- the optimal number of clusters by eigengap heuristic
	- all eigen values
	- all eigen vectors

	This method performs the eigen decomposition on a given affinity matrix,
	following the steps recommended in the paper:
	1. Construct the normalized affinity matrix: L = D−1/2ADˆ −1/2.
	2. Find the eigenvalues and their associated eigen vectors
	3. Identify the maximum gap which corresponds to the number of clusters
	by eigengap heuristic

	References:
	https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
	http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
	"""	

	plt.title('Largest eigen values of input matrix \nClusters:%d'%nb_clusters)
	plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
	plt.grid()
	#if(not Path('eigen.png').exists()):
	plt.savefig(filename)
	plt.close()
	# Identify the optimal number of clusters as the index corresponding
	# to the larger gap between eigen values
	

def calculate_gap(X,n_neighbors=10,path="",PLOT_GAP=False):
	affinity_matrix = kneighbors_graph(X, n_neighbors,
					include_self=True)
	affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)

	n_components = int(affinity_matrix.shape[0]*0.9)
	v, lambdas, connected = spectral_embedding(affinity_matrix,n_components=n_components)

	lambdas = np.sort(-lambdas)
	
	index_largest_gap = np.argmax(np.diff(lambdas))
	nb_clusters = index_largest_gap + 1

	if PLOT_GAP:
		png_file = "eigen_gap_" + str(n_neighbors) + ".png"
		png_file = str(Path(path)/png_file)

		plot_gap(lambdas,filename=png_file,nb_clusters=nb_clusters)

	return nb_clusters,connected

if __name__ == '__main__':
	
	from sklearn import datasets
	# import some data to play with
	iris = datasets.load_iris()
	X = iris.data
	calculate_gap(X,30)
	# n_neighbors = 30

	# affinity_matrix = kneighbors_graph(X, n_neighbors,
	# 				include_self=True)
	# affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)

	# n_components = int(affinity_matrix.shape[0]*0.9)
	# v, lambdas = spectral_embedding(affinity_matrix,n_components=n_components)

	# lambdas = np.sort(-lambdas)
	# nb_clusters, eigenvalues = eigen_gap(lambdas)
	# print(v.shape,lambdas.shape)
	# print(lambdas[:10])
	# print(nb_clusters)
