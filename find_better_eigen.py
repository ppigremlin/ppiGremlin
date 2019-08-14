from __future__ import print_function

import contacts as ct
import graphprocessing as gs
import clustering as cl
import graphmining as gm
import common as cm
import logging
import json
import sys

import Bio.PDB.PDBList as pdbl
from Bio.PDB import PDBParser,NeighborSearch

from pathlib import Path
from subprocess import call
import time
				  
import networkx as nx
import numpy as np
from sklearn.metrics import calinski_harabaz_score,davies_bouldin_score,silhouette_score

import eigen_gap as eg

import traceback
import os

from collections import Counter


import matplotlib.pyplot as plt
import matplotlib.cm as plcm

def plot_eigen_gap(max_gap_connected,var_comp,dataset="",d_name="",adjust=True):
	
	if len(max_gap_connected) == len(var_comp) - 1:
		var_comp = var_comp[1:]
	else:
		print("Erro em var_comp array size",sys.stderr)

	x_connected = max_gap_connected
	c1 = '#1f77b4'
	c2 = '#2ca02c'
	
	colors = list(map(lambda x: c2 if x > 0.95 else c1,var_comp))

	data = { k:v for k,v in enumerate(map(lambda x: x[0],x_connected),1) }
	k_neighbors = np.array(list((map(lambda x: x[2], x_connected))))
	
	fig, ax = plt.subplots()
	ax.set_xlabel("Number of components d of reduced matrix")
	ax.set_ylabel("Number of clusters n")
	ax.set_title("Optimal number of clusters for " + dataset + " dataset")
	
	X = list(filter(lambda x: x[1] > 0.95,zip(data.items(),var_comp)))
	X = {k:v for k,v in [j[0] for j in X]}
	
	a95 = ax.scatter(X.keys(),X.values(), marker='o',c='#1f77b4',s=50)

	X = list(filter(lambda x: x[1] < 0.95,zip(data.items(),var_comp)))
	X = {k:v for k,v in [j[0] for j in X]}

	b95 = ax.scatter(X.keys(),X.values(), marker='o',c='#ff7f0e',s=50)

	ax.grid(True)
	ax.set_axisbelow(True)
	ax.set_xlim(tuple([int(i) for i in ax.get_xlim()]))
	
	ax2 = ax.twinx()
	knb = ax2.scatter(data.keys(),k_neighbors,c="gray",marker="D")
	ax2.set_ylabel("K-neighbors")
	# if adjust:
	# 	ax.set_ylim((0,18))
	# 	ax2.set_ylim((1,500))

	ax.legend((a95,b95,knb),("d > 95%","d < 95%",'k_neighbors'))

	fig.savefig("n_clusters_" + dataset + ".png")

def plot_max_gap(x,y,y2,labels=None,show=True,save=False,fname="",path=""):

	y = [list(i) for i in zip(*y)]
	colors = list(map(lambda x: "blue" if x else "red",y[1]))
	y = y[0]

	plt.scatter(x,y,c=colors)
	plt.plot(x,y,linestyle='None')
	
	y2 = [list(i) for i in zip(*y2)]
	colors = list(map(lambda x: "blue" if x else "red",y2[1]))
	y2 = y2[0]

	plt.scatter(x,y2,c=colors)
	plt.plot(x,y2,linestyle='None')

	x_step = 1
	y_max = int(max(max(y),max(y2))*1.1+1)

	plt.xticks(np.arange(min(x), max(x)+1, x_step))
	plt.yticks(np.arange(0,y_max,x_step))
	plt.grid()

	if labels is not None:
		for i,l in enumerate(labels):
			plt.annotate(l, (x[i], y[i]))

	if save:
		if path:
			if fname is not None or fname != '':
				f_path = Path(path)/fname
			else:
				f_path = Path(path)/"test.png"
			plt.savefig(str(f_path))
		else:
			plt.savefig("test.png")
	if show:
		plt.show()

	plt.close('all')

if __name__ == '__main__': 
	sys.stdout.flush()
	interactions,int_list = ct.readInteractions("interactions.csv")
	a_types,a_type_list = ct.readAtom_Types("atom_types.csv")

	typeCode = cm.TypeCode(a_type_list,int_list)

	path = Path(sys.argv[1])
	eigen_path = path/'eigen_gap'
	pdbids_file = sys.argv[2]

	path.mkdir(parents=True,exist_ok=True)
	eigen_path.mkdir(parents=True,exist_ok=True)
	path = str(path)
	#print(path,eigen_path)

	logging.basicConfig(level=logging.DEBUG)
	
	if not (Path(path)/"graphs.txt").exists():

		logging.debug("---Read PDB ids file---")
		pdbids, chains = cm.read_pdbid_file(pdbids_file)

		logging.debug("---Read PDB files---")
		pdb_structures = cm.read_PDB_files(pdbids,directory="pdbfiles")

		logging.debug("---Write PDB chain files---")
		cm.write_pdb_files(pdb_structures,chains,directory="pdbs")

		logging.debug("---Calculate contacts---")
		contacts = ct.run_contacts(pdb_structures,chains,interactions,a_types)

		logging.debug("---Generate graphs---")
		graphs, node_labels, edge_labels = ct.gen_graphs(contacts,typeCode,path=path)

	else:

		logging.debug("---Read graphs file---")
		graphs,node_labels,edge_labels = gs.read_graphs('graphs.txt',path=path)
	
	
	##################### Generate count matrix #############################

	count_matrix_filename = "count_matrix.csv"
	if not (Path(path)/count_matrix_filename).exists():
		logging.debug("---Generate counting matrix---")

		count_matrix = gs.genCountMatrix(
					graphs,node_labels,edge_labels,typeCode,
					filename=count_matrix_filename,path=path)

	else:
		logging.debug("---Load counting matrix---")
		count_matrix = np.genfromtxt((Path(path)/count_matrix_filename), delimiter=',')


	##################### Running SVD #############################
	path = eigen_path
	
	svd = cl.SVD(count_matrix,scaler='std',with_std=False)	
	#print("Variance_comp",svd.getVarianceComposition(),len(svd.getVarianceComposition()))
	sys.stdout.flush()

	max_gap = []
	max_gap_connected = []
	
	cl_range = [i for i in range(1,count_matrix.shape[1]+1)]
	#cl_range = [i for i in range(1,10)]
	var_comp = svd.getVarianceComposition()
	with (Path(path) / 'var_comp.json' ).open(mode='w') as out_var_comp:
			json.dump(svd.getVarianceComposition(),out_var_comp,indent=4)

	for n_components in cl_range:

		# dir_path = Path(path)/("%d/"%(n_components))
		
		# dir_path.mkdir(parents=True,exist_ok=True)
		
		# status_file = dir_path/"status.csv"

		# status_file = status_file.open(mode='w')
		# status_file.write("n_components %d\n"%n_components)
		
		
		analytics = []
		neighbors = []
		
		data_matrix = svd.getReduced(n_components)
		# red_matrix_file = dir_path/'data.csv'
		# svd.saveReduced(n_components,file=red_matrix_file)


		for factor in np.arange(0.01,0.91,0.01):
		#for factor in np.arange(0.25,0.91,0.25):
			n_neighbors = int(factor*data_matrix.shape[0])
			neighbors.append(n_neighbors)
			#print(n_neighbors)
			n_cluster, connected = eg.calculate_gap(data_matrix,
											n_neighbors=n_neighbors)
			analytics.append([int(n_cluster),connected,n_neighbors])
			if connected:
				break

		max_gap.append(tuple(max(analytics)))

		connected_gaps = list(filter(lambda x: x[1],analytics))
		if len(connected_gaps):
			max_gap_connected.append(max(connected_gaps))
		else:
			max_gap_connected.append((0,True))

		
		analytics = {'analytics':analytics, 'n_neighbors':neighbors, 'n_components': n_components}

		# with (dir_path / 'analytics.json' ).open(mode='w') as out_analytics:
		# 	json.dump(analytics,out_analytics,indent=4)
		
	with (Path(path) / 'analytics.json' ).open(mode='w') as out_analytics:
			json.dump({'max_gap':max_gap, 'max_gap_connected':max_gap_connected},out_analytics,indent=4)

	with cm.cd(path):
		join_labels = True
		dataset = pdbids_file.split(".")[0]

		#plot_max_gap(cl_range,max_gap,max_gap_connected,save=True,show=False,fname="eigen_summary.png")
		plot_eigen_gap(max_gap_connected,var_comp,d_name="",dataset=dataset,adjust=False)