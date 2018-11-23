import numpy as np
import random
def get_Deg_1(Deg_V_Test):
    count = 0
    for i in range(len(Deg_V_Test)):
        if Deg_V_Test[i][1] == 1:
            count += 1
    list_1 = [0 for i in range(count)]
    list_1 = np.array(list_1)
    count1 = 0
    for i in range(len(Deg_V_Test)):
        if Deg_V_Test[i][1] == 1:
            list_1[count1] = i
            count1 +=1
    return list_1

#判断一个节点是否为度为1的节点
def in_Deg_1(Deg_V_Train,v):
    list_1 = get_Deg_1(Deg_V_Train)
    for i in range(len(list_1)):
        if list_1[i] == v:
            return 1

def dfs(graph, start, visited=None):
    if visited is None:
        visited = []
    visited.append(start)
    for next in list(set(graph[start]).difference(set(visited))):
        dfs(graph, next, visited)
    return visited

def test_connect(visited):
    if len(np.unique(visited)) == len(graph):
        return 1
    else:
        return 0

#创建List集 第i行表示 节点i的邻居节点的序号
def Create_List(Train):
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
    return List

def Create_Deg(a):
    a = np.transpose(a)
    list1 = a[0]
    list2 = a[1]
    a = np.transpose(a)
    list1 = list1.tolist()
    list2 = list2.tolist()
    list3 = list1 + list2
    list3 = np.unique(list3)
    list3 = np.array(list3)
    list3 = list3.reshape(-1, 1)
    Deg_V = [[0] for i in range(len(list3))]
    Deg_V = np.array(Deg_V)
    for i in range(len(a)):
        Deg_V[a[i][0]] += 1
        Deg_V[a[i][1]] += 1
    Deg_V_E = np.hstack((list3, Deg_V))
    Deg_V_E = np.array(Deg_V_E)
    return  Deg_V_E

E = np.loadtxt('D:/deepwalk_test/Yeast/Yeast_standard.txt',dtype=int)
list1 = list(range(0, len(E)))
number1 = (int)(len(E)*0.1)
print(number1)
number2 = (int)(len(E)*1)
slice = random.sample(list1,number2)
Test = [[0,0] for i in range(number1)]
dec = []
count= 0
Train = E

for i in range(len(slice)):
    print(count)
    if count <number1:
        E_test = E
        Train = np.delete(E_test, dec, axis=0)
        Deg_now = Create_Deg(Train)
        if in_Deg_1(Deg_now,E[slice[i]][0]) != 1 and in_Deg_1(Deg_now,E[slice[i]][1]) != 1:
            dec.append(slice[i])#需要验证slice[i]在当前最近一次形成的Train中是否包含度为1的节点
            Train = np.delete(E_test, dec, axis=0)
            graph = Create_List(Train)
            c = test_connect(dfs(graph, 0))
            if c == 0:
                dec.remove(slice[i])
            else:
                count += 1

Train = np.delete(E, dec, axis=0)
print(len(Train))
for i in range(len(dec)):
    Test[i] = E[dec[i]]
print(len(Test))
print(len(E))
np.savetxt('D:/deepwalk_test/Yeast/Train.edgelist',Train,fmt="%d %d")
np.savetxt('D:/deepwalk_test/Yeast/Test.edgelist',Test,fmt="%d %d")




'''
E_test = E
Train_test = np.delete(E_test, dec, axis=0)
print(len(Train_test))
dec.append(slice[i])
#检测度
Deg_V_Train = Create_Deg(Train_test)
E_test = E
if (in_Deg_1(Deg_V_Train, E[slice[i]][0]) != 1) and (in_Deg_1(Deg_V_Train, E[slice[i]][1]) != 1):
    Train = np.delete(E_test, dec, axis=0)
    graph = Create_List(Train)
    c = test_connect(dfs(graph, 0))
    if c == 0:
        dec.remove(slice[i])
    else:
        count += 1
'''



