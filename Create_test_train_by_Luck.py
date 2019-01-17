import numpy as np
import random

path = '/home/hongqiaochen/Desktop/Date_Link_predict/Router'

#�õ�Test�ж�Ϊ1�ĵ�
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

#�ж�һ���ڵ��Ƿ�Ϊ��Ϊ1�Ľڵ�
def in_Deg_1(Deg_V_Train,v):
    list_1 = get_Deg_1(Deg_V_Train)
    for i in range(len(list_1)):
        if list_1[i] == v:
            return 1

#��������ȱ���ͼ�ķ�ʽ����һ��ͼ
def dfs(graph, start, visited=None):
    if visited is None:
        visited = []
    visited.append(start)
    for next in list(set(graph[start]).difference(set(visited))):
        dfs(graph, next, visited)
    return visited

#���ݱ����Ľ���õ�ͼ����ͨ��
def test_connect(visited):
    if len(np.unique(visited)) == len(Deg_V_E):
        return 1
    else:
        return 0

#���ݶȵı�׼������Test��Train
def Create_Test_and_Train():
    list1 = list(range(0, len(E)))
    number1 = (int)(len(E)*0.1)
    number2 = (int)(len(E))
    slice = random.sample(list1,number2)
    Test = [[0,0] for i in range(number1)]
    dec = [[0] for i in range(number1)]
    count2 = 0
    for i in range(len(slice)):
        if count2 < number1:
            if (in_Deg_1(Deg_V_Train,E[slice[i]][0])!= 1) and (in_Deg_1(Deg_V_Train,E[slice[i]][1])!= 1 ):
                Deg_V_Train[E[slice[i]][0]][1] -= 1
                Deg_V_Train[E[slice[i]][1]][1] -= 1
                Test[count2] = E[slice[i]]
                dec[count2] = slice[i]
                count2 += 1
    Train = np.delete(E, dec, axis=0)
    return Test,Train

#���Train�Ƿ�Ϊ��ͨ�ģ����������µ���Create_Test_and_Train()
def Check_Train(Train):
    Train = np.transpose(Train)
    list1 = Train[0]
    list1 = list1.tolist()
    list2 = Train[1]
    list2 = list2.tolist()
    Train = np.transpose(Train)
    list3 = list1 + list2
    list3 = np.unique(list3)
    list3 = np.array(list3)
    list3 = list3.reshape(-1, 1)
    list4 = [[0] for i in range(len(list3))]
    list4 = np.array(list4)
    list_degree = np.hstack([list3, list4])
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
    a = test_connect(dfs(graph,0))
    return a

E = np.loadtxt(path+'/standard.txt', dtype=int)
E = np.transpose(E)
list1 = E[0]
list2 = E[1]
E = np.transpose(E)
list1 = list1.tolist()
list2 = list2.tolist()
list3 = list1 + list2
list3 = np.unique(list3)
list3 = np.array(list3)
list3 = list3.reshape(-1, 1)
Deg_V = [[0] for i in range(len(list3))]
Deg_V = np.array(Deg_V)
for i in range(len(E)):
    Deg_V[E[i][0]] += 1
    Deg_V[E[i][1]] += 1
Deg_V_E = np.hstack((list3, Deg_V))
Deg_V_E = np.array(Deg_V_E)
print(Deg_V_E)

# ��ԭ�߼��и��ڵ�Ķȸ�ֵ�� ���Լ����Դ����Լ�����ɾ��
Deg_V_Train = Deg_V_E
Test,Train = Create_Test_and_Train()
np.savetxt(path+'/Train.edgelist',Train,fmt="%d %d")
np.savetxt(path+'/Test.edgelist',Test,fmt="%d %d")



