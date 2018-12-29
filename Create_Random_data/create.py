import networkx as nx
import numpy as np

path = '/home/hongqiaochen/Desktop/Date_Link_predict/New'
M = 6000

BA = nx.random_graphs.barabasi_albert_graph(M,3)
edge_list_BA = BA.edges()
edge_arr_BA = np.array(edge_list_BA)
np.savetxt(path+'/BA.txt',edge_arr_BA,fmt='%d')
#
# RG = nx.random_graphs.random_regular_graph(3, M)
# edge_list_RG = RG.edges()
# edge_arr_RG = np.array(edge_list_RG)
# np.savetxt(path+'/RG.txt',edge_arr_RG,fmt='%d')

# ER = nx.random_graphs.erdos_renyi_graph(M, 0.002)
# edge_list_ER = ER.edges()
# edge_arr_ER = np.array(edge_list_ER)
# np.savetxt(path+'/ER.txt',edge_arr_ER,fmt='%d')
#
# WS = nx.random_graphs.watts_strogatz_graph(M, 4, 0.3)
# edge_list_WS = WS.edges()
# edge_arr_WS = np.array(edge_list_WS)
# np.savetxt(path+'/WS.txt',edge_list_WS,fmt='%d')

# adjlist = [[0 for i in range(M)]for j in range(M)]
# adjarr = np.array(adjlist)
# for i in range(len(edge_arr)):
#     adjarr[edge_arr[i][0]][edge_arr[i][1]] = 1
#     adjarr[edge_arr[i][1]][edge_arr[i][0]] = 1
# print(adjarr)
# np.savetxt(path+'/adjarr.txt',adjarr,fmt='%d')