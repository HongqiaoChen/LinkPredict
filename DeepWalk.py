import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import word2vec
import numpy as np
import random

def Start(list_v):
    choice = List[list_v[0]]
    r = random.randint(0,len(choice)-1)
    list_v[1] = choice[r]

def DFS(list_v,k,flag):
    choice = List[list_v[k-1]]
    if choice == None:
        flag += 1
        list_v[k] = list_v[k-1]
    else:
        r = random.randint(0,len(choice)-1)
        list_v[k] = choice[r]

def BFS(list_v,k,flag):
    choice = List[list_v[k-2]]
    if choice == None:
        flag += 1
        list_v[k] = list_v[k - 1]
    else:
        r = random.randint(0,len(choice)-1)
        list_v[k] = choice[r]


Train = np.loadtxt('D:/deepwalk_test/Jazz/Train.edgelist',dtype=int)
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

flag = 0
alpha = 0.5
times = 10
length = 30
walkpath = [[]for i in range(len(List)*times)]
for m in range(times):
    for i in range(len(List)):
        list_v = [0 for i in range(length)]
        list_v[0]= i
        Start(list_v)
        for k in range(2,length):
            if flag == 0 :
                a = random.random()
                if a > alpha :
                    DFS(list_v,k,flag)
                else:
                    BFS(list_v,k,flag)
            else:
                list_v[k] = list_v[k-1]
        walkpath[i+m*len(List)] = list_v
walkpath = np.array(walkpath)
np.savetxt('D:/deepwalk_test/Jazz/Jazz_walkpath.txt', walkpath, fmt="%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d ")
walkpath = word2vec.Text8Corpus('D:/deepwalk_test/Jazz/Jazz_walkpath.txt')
model = word2vec.Word2Vec(walkpath,size = 64,hs = 1,min_count = 0,window = 5,sg = 1)
model.wv.save_word2vec_format('D:/deepwalk_test/Jazz/Jazz_vector.txt')
