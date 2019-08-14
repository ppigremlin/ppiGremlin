import graphprocessing as gp
import networkx as nx
import numpy as np
import json
import sys
import os
import traceback
import logging
from common import cd
from pathlib import Path
from subprocess import call
from functools import reduce

#GSPAN_PATH = '/home/cathoud/Desktop/ppigremlin/gSpan/gSpan-64'
def gen_gSpan_entries(graphs,clusters,supports,node_labels,edge_labels,type_code,path='',gSpan_path=''):
	
	gSpanFName = 'entry.gsp'

	with cd(path):

		for key,graph_list in sorted(clusters.items(),
										key=lambda x: x[0]):
			gSpanFName='%s.gsp'%key
			gp.multigraph_to_gspan(graph_list,
					node_labels,edge_labels,type_code,gspan_fname=gSpanFName)
				
def runGSpan(graphs,clusters,supports,node_labels,edge_labels,path='',gSpan_path=''):
	
	graphs_dict = clusters
	
	gSpan_out = (Path(path)/'gSpan.txt').open(mode='w') #gSpan log
	gSpan_results = dict()
	
	#Change context 
	with cd(path):

		for key,graph_list in sorted(graphs_dict.items(),
										key=lambda x: x[0]):
			gSpanFName = str(key) + ".gsp"
			
			temp_gSpan_results = []
			for min_sup in supports:
				call([gSpan_path,"-f",gSpanFName,"-s",str(min_sup),"-o","-i"],stdout=gSpan_out)
	
				Path(gSpanFName+'.fp').rename('%s_%s.fp'%(key,min_sup))

				temp_gSpan_results.append('%s_%s.fp'%(key,min_sup))

			gSpan_results[int(key)] = temp_gSpan_results

		gSpan_results = {'results':gSpan_results, 'supports':supports}
		with open('gSpan.fp','w') as out_gspan_files:
			json.dump(gSpan_results,out_gspan_files,indent=4)

	return gSpan_results,graphs_dict

def read_gSpan_results(node_labels,edge_labels,filename="gSpan.fp",path=""):

	with (Path(path)/filename).open() as filename_list:
		result_files = json.load(filename_list)

	gSpan_results = dict()
	supports = result_files['supports']
	with cd(path):		
		for cluster,files in result_files['results'].items():
			temp_results = dict()
			for filename in files:
				key = filename.split('_')[1].split('f')[0][:-1]
				temp_results[key] = gp.gspan_to_graph(filename,node_labels,edge_labels)
				os.remove(filename)
			gSpan_results[int(cluster)] = temp_results
	return gSpan_results,supports
	
def getMaximalGraphs(clusters,file="maximal.json",path=""):	

	j_maximal = []
	maximal_graphs = []
	
	for n_cluster,cluster in clusters.items():
	
		j_temp = dict()
		temp = dict()
	
		for min_sup, graphs in sorted(cluster.items()):
	
			g =	maximal(graphs)
			temp[min_sup] = [ {"graph": j["graph"], "l_graph": j["l_graph"]} for j in g ]
			j_temp[min_sup] = ["".join([i for i in nx.generate_gml(j["graph"])]) for j in g]

		maximal_graphs.append(temp)
		j_maximal.append(j_temp)
	

	with (Path(path) / file ).open(mode='w') as j_patterns_file:
		j_patterns_file.write(json.dumps(j_maximal,indent=4))

	return maximal_graphs
		
def maximal(graphs):

	############# Filter: remove 1-vertex graphs
	graphs = [i for i in filter(lambda x: x.number_of_nodes() > 1,graphs)]

	if not graphs:
		return []

	graphs.sort(key=lambda x: -x.number_of_nodes())

	### Generate Line Graphs
	graphs = [{"graph": g, "l_graph" : gp.line_graph(g)} for g in graphs]
	
	############# Get maximals

	######## Node Split
	graphs_holder = graphs
	graphs = []
	last_num_nodes =-1
	while graphs_holder:
		num_nodes = graphs_holder[-1]["graph"].number_of_nodes()
		if  num_nodes != last_num_nodes:
			graphs.append([])
			last_num_nodes = num_nodes

		graphs[-1].append(graphs_holder.pop())

	######## Edge Split	
	graphs = [sorted(i,key = lambda x: -x['graph'].number_of_edges()) for i in graphs]

	######## Graph Subgraph Isomorphism (Level 1)
	nm = lambda x,y: x['type'] >= y['type'] and (x['type'] & y['type'])
	em = nx.isomorphism.numerical_node_match(["type"],[""])

	marked = [np.full((len(g)),False) for g in graphs]
	
	for g in range(1,len(graphs)):
		count = 0

		for i in range(len(graphs[g])):
			j_size = len(graphs[g-1])
			
			if count >= j_size:
				break

			for j in range(j_size):
				if marked[g-1][j]:
					continue

				m = nx.isomorphism.GraphMatcher(graphs[g][i]["l_graph"],graphs[g-1][j]["l_graph"],
							edge_match=em,node_match=nm)

				if m.subgraph_is_isomorphic():
					count+=1		
					marked[g-1][j] = True

				if count >= j_size:
					break
			

	graphs_holder = [np.array(g) for g in graphs]
	graphs = [g[i] for g,i in zip(graphs_holder,np.invert(np.array(marked)))]
	graphs = np.concatenate(np.flip(graphs))
	
	######## Graph Subgraph Isomorphism (Level 2)
	nm = lambda x,y: x['type'] >= y['type'] and (x['type'] & y['type'])
	em = nx.isomorphism.numerical_node_match(["type"],[""])
	
	marked = np.full((len(graphs)),False)
	count = 0
	
	for i in range(len(graphs)):
	
		for j in range(len(graphs)):
			if(i == j or marked[j]):
				continue

			if(graphs[i]["graph"].number_of_nodes() >= graphs[j]["graph"].number_of_nodes()):
				
				m = nx.isomorphism.GraphMatcher(graphs[i]["l_graph"],graphs[j]["l_graph"],
							edge_match=em,node_match=nm)

				if(m.subgraph_is_isomorphic()):
					count+=1	
					marked[j] = True

	for i,g in enumerate(graphs):
		g["graph"].graph['id'] = i

	return list(graphs[np.invert(marked)])

def filter_maximal(patterns,n=10):
	path = ""
	if isinstance(patterns,str):
		try:
			with (Path(path)/patterns).open() as p_file:
				patterns = json.load(p_file)
				patterns = [{ key:[nx.parse_gml(g) for g in v] for key,v in p.items()} for p in patterns]
				r_patterns = patterns
		except IOError as e:
			raise e

	for p in patterns:
		sups = sorted(p.keys())
		
		for key in sups:
			if len(p[key]) < 1:
				continue
			
			p[key].sort(key=lambda x:-x.number_of_edges())
			p[key].sort(key=lambda x:-x.number_of_nodes())
			p[key] = p[key][:n]
			
	return patterns

def mapGraphs(clusters,patterns,supports,path=""):

	if isinstance(patterns,str):
		try:
			with (Path(path)/patterns).open() as p_file:
				patterns = json.load(p_file)
				patterns = [{ key:[nx.parse_gml(g) for g in v] for key,v in p.items()} for p in patterns]
				r_patterns = patterns
		except IOError as e:
			raise e

	patterns = filter_maximal(patterns)
	
	clusters = [i[1] for i in sorted(clusters.items())]

	for c in clusters:
		for graphs in c:
			
			for n1,n2 in graphs.edges():
				graphs[n1][n2]["patterns"] = {k:set() for k in supports}
			
			for n in graphs.nodes():
				graphs.node[n]["patterns"] = {k:set() for k in supports}
			
			graphs.graph["l_graph"] = gp.line_graph(graphs)

	
	# map_file = (Path(path)/"p_mappings").open(mode="a")
	

	cluster_idx = 0
	for p_graphs,graphs in zip(patterns,clusters):
		for min_sup, p_graphs in sorted(p_graphs.items(),key=lambda x: float(x[0])):
			for p in p_graphs:
				
				pl = gp.line_graph(p)
				str0 = ""
				for s in p.graph['ocur']:

					g = graphs[int(s)]
					gl = g.graph["l_graph"]

					nm = lambda x,y: x['type'] >= y['type'] and (x['type'] & y['type'])

					em = nx.isomorphism.numerical_node_match(["type"],[""])
					m = nx.isomorphism.GraphMatcher(gl,pl,edge_match=em,node_match=nm)

					m_iter = m.subgraph_isomorphisms_iter()

   
					# mapped_nodes = set()
					# mapped_edges = set()
					for mapping in m_iter:

						## Mapping Nodes
						for k,v in mapping.items():

							# mapped_nodes.add(k[0])
							# mapped_nodes.add(k[1])
							# mapped_edges.add("-".join(sorted(k)))
							g[k[0]][k[1]]["patterns"][min_sup].add(p.graph['id'])
							g.node[k[0]]['patterns'][min_sup].add(p.graph['id'])
							g.node[k[1]]['patterns'][min_sup].add(p.graph['id'])
					# out_str = json.dumps([min_sup,p.graph['id'],g.graph['id'],list(mapped_nodes),list(mapped_edges)]) + "#\n"
					# map_file.write(out_str)

	return patterns

def printMaximalResults(results,file,info=None):
	with open(file,'w') as out:
		sys.stdout = out

		if info:
			for k,v in sorted(info.items()):
				print(k,v)
		n_cluster = 0
		print(type(results))
		for cluster in results:
			print(400*'#')
			print('Cluster:', n_cluster)
			support = 0.1
			for graphs in cluster:
				print(100*'-')
				print("Support: %.1f"%support, end="  ")
				print("N_graphs: ", len(graphs))
				print('{%.1f'%support)
				for g in graphs:
					print(g["graph"].graph)
					print("P_id",g["graph"].graph["id"])
					print(['N%dE%d' %(g["graph"].number_of_nodes(),
						g["graph"].number_of_edges())])
					print(g["graph"].nodes(data=True))
					print(g["graph"].edges(data=True))
				print('}')
				support += 0.1
   
			n_cluster+=1

	


	sys.stdout = sys.__stdout__

def jsonParse(clusters,patterns,atom_types,supports,type_code,typenames,path=""):

	data_dir_name = 'data'
	path = Path(path) / data_dir_name
	
	path.mkdir(parents=True,exist_ok=True)
	
	### Group Info
	graphs = []
	for k,v in sorted(clusters.items()):
		for graph in v:
			graphs.append((k,graph))

	graphs.sort(key=lambda x: int(x[1].graph['id']))

	with (path/'group-info.csv').open(mode="w") as gFile:
		gFile.write('"graph","group","pdb","chain","ligand"\n')
		for g in graphs:
			k,g = g
			g = g.graph
			gFile.write('%d,%d,"%s","%s","%s"\n'%(int(g['id']),k+1,g['pdbid'],g['source'],g['target']))


	### Graphs Files
	with open(typenames,'r') as typenames_file:
		typenames = json.load(typenames_file)

	
	typeInt = {
		'ACP' : 2,
		'ARM' : 3,
		'DON' : 5,
		'HPB' : 7,
		'POS' : 11,
		'NEG' : 13,
		"HYDROPHOBIC": 23,
		"SALT_BRIDGE": 19,
		"ARM_STACK": 17,
		"H_BOND": 29,
		"REPULSIVE": 31,
		"SS_BRIDGE": 37
	}

	bin_to_prime = {
		1 : 2,
		2 : 3,
		4 : 5,
		8 : 7,
		32 : 11,
		16 : 13,
		256 : 23,
		2048 : 19,
		128 : 17,
		512 : 29,
		1024 : 31
	}

	# colors = list(reversed(["a6cee3","1f78b4","b2df8a","33a02c",'fb9a99','e31a1c',
	# 'fdbf6f','ff7f00','cab2d6','6a3d9a','ffff99','b15928']))
	chain_color = "#7CB4BE"
	lig_color = "#9BCE91"

	node_types = dict()
	int_types = dict()
	colormap = dict()
	typeMap = dict()
	atomTypeInt = 0
	intTypeInt = 0
	count = dict()

	graph_list_file = (path/"list-graphs").open(mode='w')

	########################## Generate graph files ##########################
	graphs_path = Path("graphs")
	(path / graphs_path).mkdir(parents=True,exist_ok=True)

	for s in supports:

		dir_name = 'json_%s'%s
		dir_path = graphs_path / dir_name
		
		(path/dir_path).mkdir(parents=True,exist_ok=True)

		with cd(path/dir_path):

			j_graphs = []
			for k,v in sorted(clusters.items()):
				for graphs in v:
					keys = ['pdbid','source','id']
					g = {'data':{'nodes': [], 'links': []},
						'meta':tuple([k+1]+[graphs.graph[x] for x in keys])}
	          		
					### Nodes
					for node in graphs.nodes():
						temp_node = dict()
						temp_node['index'] = int(node)

						node = graphs.node[node]

						temp_node['patterns'] = list(node['patterns'][s])
						temp_node['chain'] = node['chain']
						temp_node['atomName'] = node['atomName']
						temp_node['residueNumber'] = str(node['residueNumber'])
						temp_node['residueName'] = node['residueName']
						temp_node['isLigand'] = True if node['isLigand'] else False
						temp_node['atomType'] = atom_types[node['residueName']][node['atomName']]
						temp_node['color'] = lig_color if temp_node['isLigand'] else chain_color

						atomTypeInt = reduce(lambda x,y: x*y,[i for i in map(lambda x: typeInt[x],temp_node['atomType'])])
						atomType = temp_node['atomType'] = '/'.join(sorted([i for i in map(lambda x: typenames[x],temp_node['atomType'])]))
						temp_node['atomTypeInt'] = str(atomTypeInt)

						g['data']['nodes'].append(temp_node)

					g['data']['nodes'].sort(key=lambda x: x['index'])

					### Edges (links)
					nodes = graphs.node
					for edge in graphs.edges(data=True):
						
						temp_edge = dict()
						temp_edge['patterns'] = list(edge[2]['patterns'][s])

						if nodes[edge[0]]["isLigand"]:
							temp_edge['source'] = int(edge[1])
							temp_edge['target'] = int(edge[0])

						else:
							temp_edge['source'] = int(edge[0])
							temp_edge['target'] = int(edge[1])

						interactionType = type_code.type(edge[2]['type'])

						temp_edge['interactionType'] = "/".join(
							typenames[t] for t in interactionType)
						temp_edge['distance'] = str(edge[2]['distance'])
						temp_edge['interactionTypeInt'] = str(reduce(lambda x,y: x*y,
								[typeInt[t] for t in interactionType]))
						g['data']['links'].append(temp_edge)
					j_graphs.append(g)

			### Write json graph files			
			for g in j_graphs:
				fname = "g%d.%s.%s.%d.graph.json"%(g['meta'])
				f_path = dir_path/fname

				graph_list_file.write(str(f_path) + '\n')
				with open(fname,'w') as out_json:
				 	out_json.write(json.dumps(g['data'],indent=4))

	graph_list_file.close()

	########################## Generate mapping files ##########################
	mappings_array = []
	
	for key,cl in sorted(clusters.items()):
		
		for sup,graphs in patterns[key].items():
	
			patterns_array = []
			for g in sorted(graphs,key=lambda x: int(x.graph["id"])):
				patterns_array.append({
					"entranceGraphs": g.graph["ocur"],
					"patternLabel" : str(g.graph['id']),
					"patternSize" : len(g)
					})
	
			mappings_array.append({
				"support" : sup,
				"patterns" : patterns_array,
				"group" : str(key+1)
				})


	with (path/"files_mapping.json").open(mode="w") as files_mapping_file:
		files_mapping_file.write(json.dumps(mappings_array,indent=4))

	########################## Generate pattern files ##########################
	patterns_list_file = (path/"list-patterns-graphs").open("w")

	dir_name = 'patterns'
	dir_path = path / dir_name
	dir_path.mkdir(parents=True,exist_ok=True)
	
	with cd(dir_path):
		for s in supports:
			for k in clusters:
				pattern_group = patterns[k]
				for graph in pattern_group[s]:
					p = {'nodes': [], 'links': [],'graphproperties':{}}

					for node in graph.nodes():
						temp_node = {}	
						temp_node['index'] = int(node)

						atomType = type_code.type(graph.nodes[node]['type'])
						atomTypeInt = reduce(lambda x,y: x*y,[typeInt[i] for i in atomType])
						atomType = '/'.join(sorted(map(lambda x: typenames[x], atomType)))
						temp_node['atomType'] = atomType

						temp_node['atomTypeInt'] = str(atomTypeInt)
						p['nodes'].append(temp_node)

					p['nodes'].sort(key = lambda x: x["index"])

					for edge in graph.edges(data=True):
						temp_edge = {}
						temp_edge['source'] = int(edge[0])
						temp_edge['target'] = int(edge[1])
						interactionType = type_code.type(edge[2]['type'])
						temp_edge['interactionType'] = "/".join(
							typenames[t] for t in interactionType)
						temp_edge['interactionTypeInt'] = str(reduce(lambda x,y: x*y,
								[typeInt[t] for t in interactionType]))
						p['links'].append(temp_edge)
						
					p['graphproperties']['inputgraphs'] = [clusters[k][int(i)].graph["id"] for i in graph.graph['ocur']]
					fname = "g%d.gsp_%s.maximal.fp.patternIndex%d.json" % (k+1,s,graph.graph["id"])

					patterns_list_file.write('patterns/'+fname+"\n")

					with open(fname,'w') as out_json:
						out_json.write(json.dumps(p,indent=4))

	patterns_list_file.close()

	########################## Vertex number file ##########################
	vert_number_file = str(path/'vert_number.csv')

	vert_number_array = [["group","support","patternSize","occurrences"]]
	for key,pattern_group in enumerate(patterns):
		for sup,graphs in patterns[key].items():
			patterns_info = {}
			for g in graphs:
				patterns_info[len(g)] = patterns_info.get(len(g),0) + 1
			#print(patterns_info)
			for p_key,p_val in sorted(patterns_info.items()):
				vert_number_array.append([key+1,sup,p_key,p_val])

	np.savetxt(vert_number_file, vert_number_array, delimiter=',',fmt="%s")

def maximalCount(patterns,node_labels,edge_labels,typenames,type_code,path=""):
	data_dir_name = 'data'
	path = Path(path) / data_dir_name
	filename = 'count_atoms_and_interactions.csv'
	fpath = str(path/filename)

	edge_single_labels = []
	i = edge_labels[0]
	for j in edge_labels:
		if(i == j):
			edge_single_labels.append(i)
			i = i << 1


	labels = { k:v for v,k in enumerate(node_labels + edge_single_labels,2)}
	columns = len(labels)

	r_edge_single_labels = [ i for i in reversed(edge_single_labels)]

	count_matrix = ["/".join(typenames[t] for t in type_code.type(l)) for l in labels]
	for i in range(len(count_matrix)):
		if i < len(node_labels):
			count_matrix[i] = "atoms" + count_matrix[i]
		else:
			count_matrix[i] = "inter" + count_matrix[i]

	count_matrix = [['group','support'] + count_matrix] 
	
	for cl_idx in range(len(patterns)):
		for min_sup,v in patterns[cl_idx].items():
			line = [cl_idx,min_sup] + [0]*columns
	
			for graph in v:
				types = nx.get_node_attributes(graph,"type").values()
				for t in types:
					line[labels[t]] += 1

				for attr in nx.get_edge_attributes(graph,"type").values():
					for t in r_edge_single_labels:
						if t <= attr:
							attr -= t
							line[labels[t]] += 1
			count_matrix.append(line)
	count_matrix = np.array(count_matrix)
	
	with Path(fpath).open(mode="w") as file:
		np.savetxt(file, count_matrix, delimiter=',',fmt="%s")