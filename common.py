import Bio.PDB.PDBList as pdbl
import numpy as np
import json
import contacts as ct
from Bio.PDB import PDBParser, PDBIO
from pathlib import Path
import os

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = Path(newPath).expanduser()

    def __enter__(self):
        self.savedPath = Path.cwd()
        os.chdir(str(self.newPath))

    def __exit__(self, etype, value, traceback):
        os.chdir(str(self.savedPath))

class TypeCode:
	def __init__(self,atom_types,int_types):
		
		type_map = enumerate(sorted(atom_types) + sorted(int_types))
		self.type_map = {v:(1<<k) for k,v in type_map}
		self.keys = sorted(self.type_map.values())
		self.rev = {k:v for v,k in self.type_map.items()}
		#print(self.type_map)

	def __getitem__(self,t):
		return self.type_map[t]


	def get_rev(self,n):
		return self.rev[n]

	def type(self,n):
		return [self.rev[i] for i in filter(lambda x: x & n,self.keys)]
	
	def code(self,n):
		return [i for i in filter(lambda x: x & n,self.keys)]

	def len(self):
		return len(self.type_map)

	def max(self):
		return 1<<(self.len()-1)

	def __str__(self):
		return self.type_map.__str__()
		
class TypeMap:
	def __init__(self,json_file):
		with open(json_file,"r") as json_file_holder:
			self.type_map = json.loads(json_file_holder.read())

	def __str__(self):
		return self.type_map.__str__()

	def __getitem__(self,t):
		return self.type_map[t]

class ChainSelect:
	def __init__(self,chains,model=0):
		self.chains = list(chains)
		self.model = model

	def accept_chain(self,chain):
		if chain.get_id() in self.chains:
			#print(chain,chain.get_id())
			return True
		return False

	def accept_model(self,model):
		if model.get_id() == self.model:
			#print("File contains multiple models")
			return True
		return False

	def accept_residue(self,residue):
		return True

	def accept_atom(self,atom):
		return True

def read_PDB_files(pdbids,directory="",show_info=True):

	path = Path.cwd() / directory
	path.mkdir(parents=True,exist_ok=True)

	# with open(pdbids) as input_pdbs:
	# 	pdbids = list(map(lambda x: x.split(","),input_pdbs.read().split("\n")))
	# 	pdbids = list(filter(lambda x: x[0][0] != "#",pdbids))

	pdb_structures = dict()
	info = dict()

	for i in pdbids:

		if (path / ("pdb%s.ent" %i.lower())).exists():
			f_name = str(path / ("pdb%s.ent" %i.lower()))

		elif (path / ("%s.pdb" %i.lower())).exists():
			f_name = str(path / ("%s.pdb" %i.lower()))

		else:
			f_name = pdbl().retrieve_pdb_file(pdb_code=i,pdir=directory,file_format='pdb')

    
		#chains = [i[2*n+1:2*n+3] for n in range(len(i)//2)]
		structure = PDBParser(QUIET=True).get_structure(i,f_name)

		pdb_structures[i] = structure
		
		num_models = len([j for j in structure.get_models()])
		if num_models > 1:
			info[i] = num_models
		
		#exit()
	#chains = {k:v for k,v in }
	# if show_info:
	# 	print("%d pdbs have multiple models:"%len(info))
	# 	for k,v in sorted(info.items()):
	# 		print("%s-(%d)"%(k,v))
	# 	print("Total number of models: %s"%sum(info.values()))

	return pdb_structures	

def read_pdbid_file(filename):
	chains = dict()
	with open(filename,"r") as input_pdbs:
		chains = list(map(lambda x: x.split(","),input_pdbs.read().split("\n")))
		chains = list(filter(lambda x: x[0][0] != "#",chains))
		chains = [[c[0]] + [(c[2*n+1],c[2*n+2]) for n in range(len(c)//2)] for c in chains]
		
		chains = {k:v for k,v in [(i[0],i[1:]) for i in chains] }

	pdbids = sorted(chains.keys())

	return pdbids,chains

def gen_primes(n):
	primes = []

	number = 2

	while n > 0:
		is_prime = True
		#print(range(2,number))

		for p in range(2,number):
			if not(number % p):
				is_prime = False
				break
		if is_prime:
			primes.append(number)
			n -= 1
		number += 1
	return primes

def write_pdb_files(structures,chains,directory=""):

	path = Path(".") / directory
	path.mkdir(parents=True,exist_ok=True)
	#print(path)

	#print("write pdb files:")
	for pdbid,pdb_structure in structures.items():
		#print(pdbid)	
		# models = map(lambda x: x.get_id(),pdb_structure.get_models())
		io = PDBIO()
		io.set_structure(pdb_structure)
		# for models in models:
		for s,t in chains[pdbid]:
			#print(pair)
			fname = str(path/("%s.%s.pdb" %(pdbid,s)))
			io.save(fname,ChainSelect([s,t]))

def read_clusters(clusters,graphs,path=""):
	if isinstance(clusters,str):
		try:
			file = (Path(path)/clusters).open()
			clusters = np.loadtxt(file,dtype=int,delimiter=",")
		except IOError as e:
			raise e
	#else:
	
	#clusters = []
	graphs_dict = dict()

	#Construct dictionary structure
	for key,graph in zip(clusters,graphs):
		graphs_dict[key] = graphs_dict.get(key,[]) + [graph]

	return graphs_dict

def fill_label_set(label_set):
	max_label = max(label_set)
	i = 1

	new_labels = set(label_set)
	while i < max_label:
		new_labels.add(i)
		i*=2
	return sorted(list(new_labels))
			
if __name__ == '__main__':
	###################### TypeCode Test
	interactions,int_list = ct.readInteractions("interactions.csv")
	a_types,a_type_list = ct.readAtom_Types("atom_types.csv")

	mask = TypeCode(a_type_list,int_list)

	print(mask.max())

	print(mask.type(384))
	print(mask.code(384))
	print(mask.get_rev(256))
	###################### TypeMap test
	#print(TypeMap("typenames.json"))

	###################### GenPrimes test
	#print(gen_primes(10))

	###################### Read tests 
	# pdbids, chains = read_pdbid_file("pdbs_test.txt")

	# pdb_structures = read_PDB_files(pdbids,directory="pdbfiles")

	# write_pdb_files(pdb_structures,chains,directory="pdbs")

	#

	#print(sorted(pdb_structures.items()))
	#read_PDB_files("pdbIdsBackup.txt",'pdbfiles')
	# with open(sys.argv[i+6],"r") as input_pdbs:
	# 		#pdbids = list(map(lambda x: x.split(" "),input_pdbs.read().split("\n")))
	# 		#pdbids = list(map(lambda x: [x[0],x[1].split(",")],pdbids))
	# 		pdbids = list(map(lambda x: x.split(","),input_pdbs.read().split("\n")))
	# 		#print(pdbids)
	# 		pdbids = list(filter(lambda x: x[0][0] != "#",pdbids))
	# 		#print(pdbids)
	# 		pdb_counter = len(pdbids)
	# 		if log_print:
	# 			print("PDB Id's found: %d" %len(pdbids))
	# 			print(pdbids)
	# 			print("PDB Id's found: %d" %len(pdbids),file=log_file)
	# 			print(pdbids,file=log_file)
	# 		t = time.clock()	
	# 		for i in pdbids:
	# 			try:   
	# 				print("Downloading PDB ID %s" % i[0])
	# 				f_name = pdbl().retrieve_pdb_file(pdb_code=i[0],pdir="pdbfiles",file_format='pdb')
					
	# 				#f_name = "pdbfiles/pdb%s.ent" % i[0].lower()
	# 				#print(f_name)
	# 				chains = [i[2*n+1:2*n+3] for n in range(len(i)//2)]

	# 				pdb_structure = PDBParser(QUIET=True).get_structure(i[0],f_name)
	# 				print("Contacts",i[0],	chains)
	# 				num_chains += len(chains)
	# 				contacts = ct.calc_contacts(pdb_structure,chains,interactions,a_types)
	# 				if parameters["multigaph"]:
	# 					graph_counter+= len(ct.gen_multi_graph(contacts,ID=i,
	# 						ofilemode="a",ofilename=gfilename))
	# 				else:
	# 					graph_counter+= len(ct.gen_graph(contacts,ID=i,
	# 						ofilemode="a",ofilename=gfilename))
	# 			except BaseException as e:
	# 				traceback.print_exc(file=sys.stderr)
	# 				print(str(e))
	# 				print("Erro")
      
				
	# 		t = time.clock() - t
	# 		print("Time:",t)
	# 		print("Time:",t,file=log_file)


	# 	print("Graphs Generated:",graph_counter)
	# 	print("Graphs Generated:",graph_counter,file=log_file)	