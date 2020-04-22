from pathlib import Path
import networkx as nx
import numpy as np
import common as cm
import copy
import pdb
import sys

def print_graph(g):
	print("G_Id: ",g.graph['title'],"Id:",g.graph['id'])
	print("Component: N%i E%i" %(g.number_of_nodes(),g.number_of_edges()))
	print(g.edges.data('type'))
	print("Edges:",g.edges)

	...

def read_graphs(filename,path=""):

	graphs = []
	#Read graphs file	
	with cm.cd(path):
		with open(filename,"r") as input_graphs:
			str_graphs = input_graphs.read().split("#")[:-1]

	#Generate graphs
	graphs = [nx.parse_gml(i) for i in str_graphs]

	#Generate node and edge label sets
	node_labels = set()
	edge_labels = set()

	for g in graphs:
		for label in list(nx.get_node_attributes(g,'type').values()):
			node_labels.add(label)
		for label in list(nx.get_edge_attributes(g,'type').values()):
			edge_labels.add(label)

	node_labels = sorted(list(node_labels))
	edge_labels = sorted(list(edge_labels))
	
	edge_labels = cm.fill_label_set(edge_labels)
	node_labels = cm.fill_label_set(node_labels)
	
	for g in graphs:
		g.graph['node_map'] = {k:v for v,k in enumerate(sorted(g.nodes()))}
	
	return graphs,node_labels,edge_labels

def genCountMatrix(graphs,node_labels,edge_labels,typeCode,matrix_type="full",method="simple",join_labels=True,filename="count_matrix.csv",path=""):
	
	node_labels = {k:v for v,k in enumerate(node_labels)}
	matrix_label_set = set()
	count_array = np.empty([len(graphs)],dtype=object)
	count_graph = 0

	if method == 'path':
		for g in graphs:
			shortPaths = [i for i in nx.all_pairs_shortest_path_length(g)]
			shortPaths = {i[0]:i[1] for i in shortPaths}

			temp = dict()
			for i in g.nodes:
				
				for j in g.nodes:
					if i >= j:
						continue
					
					labels = [g.node[i]["type"],g.node[j]["type"]]
					shortest_length = shortPaths[i][j]

					if join_labels:
						labels = [[l] for l in labels]
					else:
						labels = [mask.code(l) for l in labels]
					
					for l1 in labels[0]:
						for l2 in labels[1]:
							temp_labels = sorted(map(lambda x: node_labels[x],[l1,l2]))
							
							label = "%d-%d-%d"%(temp_labels[0],temp_labels[1],shortest_length)
							matrix_label_set.add(label)

							if matrix_type == "full":
								temp[label] = temp.get(label,0) + 1
							elif matrix_type == "bin":
								temp[label] = temp.get(label,1)

			count_array[count_graph] = temp
			count_graph += 1
	else:
		for g in graphs:
			temp = dict()
			for e in g.edges():
				labels = [g.node[e[0]]["type"],g.node[e[1]]["type"]]
				
				#### Added after join labels 
				if join_labels:
					labels = [[l] for l in labels]
				else:
					labels = [mask.code(l) for l in labels]
				
				for l1 in labels[0]:
					for l2 in labels[1]:
						temp_labels = sorted(map(lambda x: node_labels[x],[l1,l2]))
					
						label = "%d-%d"%(temp_labels[0],temp_labels[1])
						matrix_label_set.add(label)

						if matrix_type == "full":
							temp[label] = temp.get(label,0) + 1
						elif matrix_type == "bin":
							temp[label] = temp.get(label,1)

			count_array[count_graph] = temp
			count_graph += 1

	matrix_label_set = [i for i in matrix_label_set]
	matrix_label_set.sort()
	
	count_matrix = np.zeros([len(graphs),len(matrix_label_set)],dtype=int)

	for i in range(len(count_array)):
		for j in range(len(matrix_label_set)):			
			count_matrix[i][j] = count_array[i].get(matrix_label_set[j],0)
	
	ct_mtx_path = Path(path)/filename
	np.savetxt(ct_mtx_path,count_matrix,fmt='%d', delimiter=",")

	return count_matrix	

def multigraph_to_gspan(graphs,node_labels,edge_labels,mask,gspan_fname="entry.gsp"):
	# print("Debug multigraph_to_gspan")
	# print('Edge labels')
	# print(edge_labels)
	edge_labels = cm.fill_label_set(edge_labels)
	# print('New edge labels')
	# print(edge_labels)
	node_labels = cm.fill_label_set(node_labels)
	graphs_holder = graphs
	graphs = []
	node_map = None
	node_labels = {k:v for v,k in enumerate(node_labels)}
	edge_labels = {k:v for v,k in enumerate(edge_labels)}
	
	for i in range(len(graphs_holder)):
		
		node_map = {k:v for v,k in enumerate(sorted(graphs_holder[i].nodes()))}
		graphs.append(copy.deepcopy(graphs_holder[i]))
		graphs[i].graph["node_map"] = node_map
	
		nx.relabel_nodes(graphs[i],node_map,copy=False)

	temp_str = ""

	for i in range(len(graphs)):

		temp_str1 = ["t # " + str(i) + "\n"]
		nodes = sorted(graphs[i].nodes)

		for j in nodes:
			if node_labels:
				node_type = graphs[i].node[j]['type']
				temp_str1 += ["v %d %d\n" % (j,node_labels[node_type])]
			else:
				temp_str1 += ["v %d\n" % (j)]
			
		for j in graphs[i].edges(data=True):
			ml = len(mask.code(j[2]["type"]))
			for l in mask.code(j[2]["type"]):
				#print(j[0],j[1],l)
				temp_str1 += ["e %d %d %d\n" % (j[0],j[1],edge_labels[l])]
				
		temp_str += "".join(temp_str1)
		count = 0
		
	with open(gspan_fname,"w") as gspan_file:
		gspan_file.write(temp_str)

	node_map = [{k:v for v,k in enumerate(sorted(g.nodes()))} for g in graphs_holder]
	return node_map		
      
def gspan_to_graph(filename,node_labels,edge_labels,multigraph=False):

	with open(filename,"r") as file:
		patterns = [ [line for line in group.split("\n")] for group in file.read().split("\n\n")]
	graphs = []
	
	node_labels = {k:v for k,v in enumerate(node_labels)}
	edge_labels = {k:v for k,v in enumerate(edge_labels)}

	for i in patterns:
		if len(i) < 2:
			continue
		i[0] = i[0].split()

		temp = None
		if(multigraph):
			temp = nx.MultiGraph()
		else:
			temp = nx.Graph()
		for j in i[1:]:

			j = j.split()
			if j[0] == "v":
				temp.add_node(int(j[1]),type=node_labels[int(j[2])])
			if j[0] == "e":
				if(temp.has_edge(int(j[1]),int(j[2]))):
					temp[int(j[1])][int(j[2])]['type'] += edge_labels[int(j[3])]
				else:
					temp.add_edge(int(j[1]),int(j[2]),type=edge_labels[int(j[3])])
			if j[0] == "x":
				temp.graph["ocur"] = j[1:]
		
		temp.graph['support'] = i[0][4]
		temp.graph['id'] = i[0][2]
		
		graphs.append(temp)
	
	return graphs

def line_graph(g):

	#### Create line_graph
	gn = nx.Graph(nx.line_graph(g))

	#### Labeling Edges
	for n1,n2 in gn.edges():
		if n1[0] == n2[0]:
			gn[n1][n2]["type"] = g.nodes[n1[0]]["type"]
		elif n1[0] == n2[1]:
			gn[n1][n2]["type"] = g.nodes[n1[0]]["type"]
		else:
			gn[n1][n2]["type"] = g.nodes[n1[1]]["type"]
		gn[n1][n2]["type"] = 0

	#### Labeling Nodes
	for n in gn.nodes():
		gn.nodes[n]['type'] = g[n[0]][n[1]]['type']

	return gn