import numpy as np
import pandas as pd
import random
from gensim.models import word2vec

def CN_Similarity(V1,V2):
    CN = list(set(l[V1]).intersection(set(l[V2])))
    return len(CN)

def RA_Similarity(V1,V2):
    CN =list(set(l[V1]).intersection(set(l[V2])))
    S = 0
    for i in range(len(CN)):
        S = 1/len(l[CN[i]]) + S
    return S

def AA_Similarity(V1,V2):
    CN =list(set(l[V1]).intersection(set(l[V2])))
    S = 0
    for i in range(len(CN)):
        S = 1/np.log(len(l[CN[i]])) + S
    return S

def Start(list_v):
    choice = l[list_v[0]]
    r = random.randint(0,len(choice)-1)
    list_v[1] = choice[r]

def DFS(list_v,k,flag):
    choice = l[list_v[k-1]]
    if choice == None:
        flag += 1
        list_v[k] = list_v[k-1]
    else:
        r = random.randint(0,len(choice)-1)
        list_v[k] = choice[r]

def BFS(list_v,k,flag):
    choice = l[list_v[k-2]]
    if choice == None:
        flag += 1
        list_v[k] = list_v[k - 1]
    else:
        r = random.randint(0,len(choice)-1)
        list_v[k] = choice[r]

def Deepwalk(p,times,length,window,min_count,alpha):
    flag = 0
    walkpath = [[] for i in range(len(l) * times)]
    for m in range(times):
        for i in range(len(l)):
            list_v = [0 for i in range(length)]
            list_v[0] = i
            Start(list_v)
            for k in range(2, length):
                if flag == 0:
                    a = random.random()
                    if a > p:
                        DFS(list_v, k, flag)
                    else:
                        BFS(list_v, k, flag)
                else:
                    list_v[k] = list_v[k - 1]
            walkpath[i + m * len(l)] = list_v
    walkpath = np.array(walkpath)
    np.savetxt('/home/hongqiaochen/Desktop/Link_predict/USAir/USAir_walkpath.txt', walkpath, fmt='%d',
               delimiter=' ')
    walkpath = word2vec.Text8Corpus('/home/hongqiaochen/Desktop/Link_predict/USAir/USAir_walkpath.txt')
    model = word2vec.Word2Vec(walkpath, size=64, hs=1, min_count=min_count, window=window, sg=1, alpha=alpha)
    model.wv.save_word2vec_format('/home/hongqiaochen/Desktop/Link_predict/USAir/USAir_vector.txt')
    V = np.loadtxt('/home/hongqiaochen/Desktop/Link_predict/USAir/USAir_vector.txt', dtype=float, skiprows=1)
    V = V[np.lexsort(V[:, ::-1].T)]
    V = np.delete(V, 0, axis=1)
    return V

def DW_Similarity(V,V1,V2):
    temp = np.sqrt(np.sum(np.square(V[V1] - V[V2])))
    S = float(1/(1+temp))
    return S




Test = np.loadtxt('/home/hongqiaochen/Desktop/Link_predict/USAir/Test.edgelist',dtype=int)
Train = np.loadtxt('/home/hongqiaochen/Desktop/Link_predict/USAir/Train.edgelist',dtype=int)
E = np.loadtxt('/home/hongqiaochen/Desktop/Link_predict/USAir/USAir_standard.txt',dtype=int)
# 创建Not集
P_E = 0
for i in range(len(E)):
    if E[i][0] > P_E:
        P_E = E[i][0]
    elif E[i][1] > P_E:
        P_E = E[i][1]
P_E += 1
N = [[1]*P_E for i in range(P_E)]
N = np.array(N)
for i in range(0,len(E)):
    a = E[i][0]
    b = E[i][1]
    N[a][b] = 0
for i in range(0,P_E):
    N[i][i] = 0
count2 =0
for i in range(0,P_E):
    for j in range(0,P_E):
        if N[i][j] == 1:
            count2 += 1
number = count2
i = 0
j = 0
count3 = 0
NotA = [0 for i in range(number)]
NotA = np.array(NotA)
NotB = [0 for i in range(number)]
NotB = np.array(NotB)
for i in range(0,P_E):
    for j in range(0,P_E):
        if N[i][j] == 1:
            NotA[count3] = i
            NotB[count3] = j
            count3 += 1
Not = np.vstack([NotA,NotB])
Not=np.transpose(Not)
# 创建List集 :List1和List2_代替邻接矩阵
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
list_degree1 = np.hstack([list3,list4])
list_degree2 = np.hstack([list3,list4])
for i in range(len(Train)):
    list_degree1[Train[i][0]][1] += 1
for i in range(len(Train)):
    list_degree2[Train[i][1]][1] += 1
List1 = [[] for i in range(len(list_degree1))]
for i in range(len(Train)):
    List1[Train[i][0]].append(Train[i][1])
for i in range(len(list_degree1)):
    List1[i] = list(set(List1[i]))
List2 = [[] for i in range(len(list_degree2))]
for i in range(len(Train)):
    List2[Train[i][1]].append(Train[i][0])
for i in range(len(list_degree2)):
    List2[i] = list(set(List2[i]))
# 任意一个点的邻居节点
l = [[] for i in range(len(list_degree1))]
for i in range(len(list_degree1)):
    l1 = List1[i]
    l2 = List2[i]
    l[i] = list(set(l1).union(set(l2)))
# 构建所有可能的边 及其对应的连通性
Point = np.unique(E)
Num = len(Point)
Adjacency = [[0]*Num for i in range(Num)]
Adjacency = np.array(Adjacency)
for i in range(len(E)):
    Adjacency[E[i][0]][E[i][1]] = 1
    Adjacency[E[i][1]][E[i][0]] = 1
for i in range(Num):
    Adjacency[i][i] = -1
U = [[0,0,0] for i in range(Num*Num)]
count = 0
for i in range(Num):
    for j in range(Num):
        U[count][0] = i
        U[count][1] = j
        U[count][2] = Adjacency[i][j]
        count += 1
U = np.array(U)
det = []
for i in range(len(U)):
    if U[i][2] == -1:
        det.append(i)
U = np.delete(U,det,axis=0)
u = np.delete(U,2,axis=1)
c = np.delete(U,[0,1],axis=1)
for i in range(len(u)):
    if u[i][0]>u[i][1]:
        temp = u[i][0]
        u[i][0] = u[i][1]
        u[i][1] = temp
U = np.hstack((u,c))
length = len(np.unique(U))
list1 = [[0]*length for i in range(length)]
list1 = np.array(list1)
for i in range(len(U)):
    list1[U[i][0]][U[i][1]] +=1
count = 0
for i in range(length):
    for j in range(length):
        if list1[i][j] > 0 :
            count += 1
list2 = [[0,0,0] for i in range(count)]
list2 =np.array(list2)
count = 0
for i in range(length):
    for j in range(length):
        if list1[i][j] > 0:
            list2[count][0] = i
            list2[count][1] = j
            list2[count][2] = Adjacency[i][j]
            count += 1
U = list2
count1 = 0
for i in range(len(U)):
    if U[i][2] == 1:
        count1 +=1
u = np.delete(U,2,axis=1)
c = np.delete(U,[0,1],axis=1)
total_edge = [[0,0] for i in range(len(u))]
for i in range(len(u)):
    total_edge[i]=[u[i][0],u[i][1]]
connect = [0 for i in range(len(c))]
for i in range(len(c)):
    connect[i] = c[i][0]



# 开始计算相似度
S_CN = [0 for i in range(len(total_edge))]
for i in range(len(u)):
    S_CN[i] = CN_Similarity(total_edge[i][0],total_edge[i][1])

S_RA = [0 for i in range(len(total_edge))]
for i in range(len(u)):
    S_RA[i] = RA_Similarity(total_edge[i][0],total_edge[i][1])

S_AA = [0 for i in range(len(total_edge))]
for i in range(len(u)):
    S_AA[i] = AA_Similarity(total_edge[i][0],total_edge[i][1])

S_DW = [0 for i in range(len(total_edge))]
V = Deepwalk(p=0, times=5, length=20, window=10,min_count=0, alpha=0.1)
for i in range(len(u)):
    S_DW[i] = DW_Similarity(V,total_edge[i][0],total_edge[i][1])

data = pd.DataFrame({'total_edge':total_edge,'connect':connect,'S_CN':S_CN,'S_RA':S_RA,'S_AA':S_AA,'S_DW':S_DW})
print(data)
data.to_csv('/home/hongqiaochen/Desktop/Link_predict/USAir/mix_4.csv',index=False)
