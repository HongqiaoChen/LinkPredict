import networkx as nx
import numpy as np


path = '/home/hongqiaochen/Desktop/Date_Link_predict/USAir'

E = np.loadtxt(path+'/standard.txt',dtype=int)
G = nx.Graph()
G.add_edges_from(E)
#得到G[1]的邻接表（无权）
List = list(dict(G[3]).keys())
print(List)

