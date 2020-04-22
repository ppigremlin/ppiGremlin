import sys, numpy as np
import logging
import networkx as nx
import common as cm

from Bio.PDB import NeighborSearch
from pathlib import Path

def printDict(d):
	for k,v in d.items():
		print(k)
		for x,y in v.items():
			print(x,y)

def dist3D(a,b):
	return (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2 

def readAtom_Types(f_name):
	try:
		f_handle = open(f_name,"r")
	except IOError as e:
		
		print("I/O error: can't open atom types file.")
		sys.exit()

	types = set()
	atm_data = dict()

	temp = []
	res_name = ""
	for line in f_handle:
		temp = line.strip('\n').split(",")
		if res_name != temp[0]:
			atm_data[temp[0]] = dict()
			res_name = temp[0]
		atm_data[res_name][temp[1]] = [ i for i in temp[2:]]
		types |= set(temp[2:])
	f_handle.close()

	types = sorted(types)
	return atm_data,types

def __str__Atom_Types(atm_data):

	temp = ""
	for k, v in sorted(atm_data.items()):
		temp += "\n" + k.__str__()
	
		for i, j in sorted(v.items()):
			temp += "\n %6s |" % (i.__str__())
			for typ in sorted(j):
				temp += "%4s" % typ
	
	return temp

def printAtom_Types(atm_data):
	print(__str__Atom_Types(atm_data))

def readInteractions(f_name):
	try:
		f_handle = open(f_name,"r")
	except IOError as e:
		print("I/O error: can't open interactions file.")
		sys.exit()


	interactions = []
	int_file = open(f_name,"r")
	for line in int_file:
		interactions.append(line.strip().split(sep=","))
		interactions[-1][3] = (float(interactions[-1][3]))**2
		interactions[-1][4] = (float(interactions[-1][4]))**2
		if(interactions[-1][1] > interactions[-1][2]):
			temp = interactions[-1][1]
			interactions[-1][1] = interactions[-1][2]
			interactions[-1][2] = temp

	f_handle.close()
	int_list = sorted(set([i[0] for i in interactions]))

	return interactions,int_list

def __str__interactions(interactions):
	temp = ""
	for i in interactions:
		temp += "%-12s| %s-%s | %5.2f -%6.2f\n" % (i[0],i[1],i[2],i[3],i[4]) 
	return temp

def printInteractions(interactions):
	print(__str__interactions(interactions))

def calc_contacts(pdb_obj,chains,interactions,atom_type,id="",ignore_models=True):
	
	if(ignore_models): 
		logging.warning("Ignoring multiple models in calc_contacts!")
	
	contacts = dict()

	for model in pdb_obj.get_models():
		
		model_contacts = dict()
		
		for chain1,chain2 in chains:
			atom_contacts = []
			
			####### Chain1
			atom_list1 = list(model[chain1].get_atoms())
			
			####### Chain2
			atom_list2 = list(model[chain2].get_atoms())
			
			####### Neighborhood Search data structure
			tree = NeighborSearch(atom_list2)
			for s_atom in atom_list1:
				if s_atom.parent.id[0] != " " or s_atom.name == "OXT":
					continue

				s_types = atom_type[s_atom.parent.resname].get(s_atom.name,[])

				if not s_types:
					continue
				
				for d_atom in tree.search(s_atom.coord,6,level="A"):
					
					if d_atom.parent.id[0] != " " or d_atom.name == "OXT":
						continue
					
					d_types = atom_type[d_atom.parent.resname].get(d_atom.name,[])
					
					if not d_types:
						continue
		
					for i in s_types:
						for j in d_types:
							count = 0
							for interaction in interactions:
								if sorted([i,j]) == sorted(interaction[1:3]):
									d = dist3D(s_atom.coord,d_atom.coord)
									if interaction[3] <= d <= interaction[4]:
										s = s_atom.serial_number
										contact = [s_atom,d_atom,interaction[0],i,j,d**0.5,s_types,d_types]
										
										atom_contacts.append(tuple(contact))
			
		#/////////////////////////////////////////////////////////////////////////////////////////////////////
		###### Output
			dtype = [('S_ATOM',object),('D_ATOM',object),('INTERACTION','U12'),('I_TYPE','U3'),
					('J_TYPE','U3'),('DISTANCE',float),('S_TYPES',object),('D_TYPES',object)]
			model_contacts_temp = np.array(atom_contacts,dtype=dtype)
			model_contacts[(chain1,chain2)] = model_contacts_temp

		contacts[model.get_id()] = model_contacts

		if ignore_models:
			break

	return contacts

def run_contacts(structures,chains,interactions,atom_type):
	contacts = dict()
	for pdbid in sorted(structures.keys()):
		contacts[pdbid] = calc_contacts(structures[pdbid],chains[pdbid],interactions,atom_type)

	return contacts

def gen_graphs(contacts,typeCode,gfilename="graphs.txt",path=""):

	graphs = []

	for pdbid, contacts in sorted(contacts.items()):
		graphs += gen_graph(contacts,pdbid,typeCode)

	for i in range(len(graphs)):
		graphs[i].graph["id"] = i

	graphs_string = ""
	graphs_string =  "".join(["".join(i for i in nx.generate_gml(g)) + "#" for g in graphs])

	#Write graphs file
	with (Path(path)/gfilename).open(mode="w") as gfile:
		gfile.write(graphs_string)

	#Generate node and edge label sets
	node_labels = set()
	edge_labels = set()
	for g in graphs:
		node_labels |= set(nx.get_node_attributes(g,'type').values())
		edge_labels |= set(nx.get_edge_attributes(g,'type').values())
		
	node_labels = sorted(node_labels)
	edge_labels = sorted(edge_labels)

	edge_labels = cm.fill_label_set(edge_labels)
	node_labels = cm.fill_label_set(node_labels)
	
	return graphs, node_labels, edge_labels

def gen_graph(contacts,pdbid,typeCode):
	
	temp = []
	
	graphs_string = ""
	g_id=0
	
	for model_idx,model in sorted(contacts.items()):
		for chains,group in sorted(model.items()):
		#/////////////////////////////////////////////////////////////////////////////////////////////////////
		###### Graph Generation
			
			g = nx.Graph()
			
			for contact in group:
				#Atom 1
				full_id = contact[0].get_full_id()
				s_data = [('model',full_id[1]),('chain',full_id[2]),("isLigand", False),('residueNumber',full_id[3][1]),
							('residueName',contact[0].get_parent().get_resname()),("atomName",full_id[4][0])]
				s_atom = full_id

				#Atom 2
				full_id = contact[1].get_full_id()
				d_data = [('model',full_id[1]),('chain',full_id[2]),("isLigand", True),('residueNumber',full_id[3][1]),
							('residueName',contact[1].get_parent().get_resname()),("atomName",full_id[4][0])]
				d_atom = full_id

				#Add or Update Edge
				if(g.has_edge(s_atom,d_atom)):
					g[s_atom][d_atom]['type'] |= typeCode[contact[2]]
				else:
					g.add_edge(s_atom,d_atom,type=typeCode[contact[2]],distance=contact[5])
								
				node = g.nodes[s_atom]
				node["type"] = node.get('type',0) | typeCode[contact[3]]
				node.update(s_data)
				
				node = g.nodes[d_atom]
				node.update(d_data)
				node["type"] = node.get('type',0) | typeCode[contact[4]]

			###########Compute Components
			number_comp = nx.number_connected_components(g)
			serial = None
			
			for comp in nx.connected_components(g):
				g_temp = g.subgraph(comp)
				g_temp = nx.Graph(g_temp)				
				g_temp = nx.convert_node_labels_to_integers(g_temp)
				
				g_temp.graph['id'] = g_id
				g_id+=1

				g_temp.graph['pdbid'] = pdbid

				g_temp.graph['source'],g_temp.graph['target'] = chains

				temp.append(g_temp)
						
	return sorted(temp,key=lambda g: g.graph['id'])