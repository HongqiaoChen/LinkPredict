import numpy as np
import operator
Train = np.loadtxt('D:/deepwalk_test/test/Train.edgelist',dtype=int)
#创建List集 第i行表示 节点i的邻居节点的序号
Train = np.transpose(Train)
list1 = Train[0]
list1 = list1.tolist()
list2 = Train[1]
list2 = list2.tolist()
Train = np.transpose(Train)
list3 = list1+list2
list3 = np.unique(list3)
list3 = np.array(list3)
list3 = list3.reshape(-1,1)
list4 = [[0] for i in range(len(list3))]
list4 = np.array(list4)
list_degree = np.hstack([list3,list4])
for i in range(len(Train)):
    list_degree[Train[i][0]][1] += 1
    list_degree[Train[i][1]][1] += 1
List = [[] for i in range(len(list_degree))]
for i in range(len(Train)):
    List[Train[i][0]].append(Train[i][1])
    List[Train[i][1]].append(Train[i][0])
for i in range(len(list_degree)):
    List[i] = list(set(List[i]))
    List[i] = list(set(List[i]))
count = 0
for i in range(len(List)):
    count += len(List[i])

graph = List

def dfs(graph, start, visited=None):
    if visited is None:
        visited = []
    visited.append(start)
    for next in list(set(graph[start]).difference(set(visited))):
        dfs(graph, next, visited)
    return visited
visited = dfs(graph,0)
if len(np.unique(visited)) == len(List):
    print('connect')
else:
    print('wrong')

