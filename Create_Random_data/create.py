import networkx as nx
import numpy as np

path = '/home/hongqiaochen/Desktop/Date_Link_predict/New'
M = 3000

# BA = nx.random_graphs.barabasi_albert_graph(M,18)
# edge_list_BA = BA.edges()
# edge_arr_BA = np.array(edge_list_BA)
# np.savetxt(path+'/BA.txt',edge_arr_BA,fmt='%d')

# RG = nx.random_graphs.random_regular_graph(80, M) #10个10个的加
# edge_list_RG = RG.edges()
# edge_arr_RG = np.array(edge_list_RG)
# np.savetxt(path+'/RG.txt',edge_arr_RG,fmt='%d')
# #6000 0.0015
# ER = nx.random_graphs.erdos_renyi_graph(M, 0.0266)
# edge_list_ER = ER.edges()
# edge_arr_ER = np.array(edge_list_ER)
# # 0.0033
# np.savetxt(path+'/ER.txt',edge_arr_ER,fmt='%d')

# 6000 4 0.3
WS = nx.random_graphs.watts_strogatz_graph(M, 80, 0.3)
edge_list_WS = WS.edges()
edge_arr_WS = np.array(edge_list_WS)
np.savetxt(path+'/WS.txt',edge_list_WS,fmt='%d')

# adjlist = [[0 for i in range(M)]for j in range(M)]
# adjarr = np.array(adjlist)
# for i in range(len(edge_arr)):
#     adjarr[edge_arr[i][0]][edge_arr[i][1]] = 1
#     adjarr[edge_arr[i][1]][edge_arr[i][0]] = 1
# print(adjarr)
# np.savetxt(path+'/adjarr.txt',adjarr,fmt='%d')