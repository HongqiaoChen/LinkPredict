import numpy as np
import random

def Similarity(V1,V2):
    temp = np.sqrt(np.sum(np.square(V1 - V2)))
    S = float(1/(1+temp))
    return S


def AUC(rate):
    Number_Sample_Test = int(len(Test) * rate)  # 伯努利分布，取样去估计总体。现在样本为总体的
    print(Number_Sample_Test)
    Number_Sample_Not = int(len(Not) * rate)  # 伯努利分布，取样去估计总体。现在样本为总体的
    print(Number_Sample_Not)
    Sample_Test = [[0, 0] for i in range(Number_Sample_Test)]
    Sample_Not = [[0, 0] for i in range(Number_Sample_Not)]
    r1 = []
    r1 = random.sample(range(0, len(Test)), Number_Sample_Test);
    r2 = []
    r2 = random.sample(range(0, len(Not)), Number_Sample_Not);
    for i in range(0, Number_Sample_Test):
        Sample_Test[i] = Test[r1[i]]
    for i in range(0, Number_Sample_Not):
        Sample_Not[i] = Not[r2[i]]
    n = len(Sample_Test) * len(Sample_Not)
    s1 = [[0.0] for i in range(Number_Sample_Test)]
    s1 = np.array(s1)
    s2 = [[0.0] for i in range(Number_Sample_Not)]
    s2 = np.array(s2)
    for i in range(0, Number_Sample_Test):
        s1[i] = Similarity(V[Sample_Test[i][0]], V[Sample_Test[i][1]])
    for j in range(0, Number_Sample_Not):
        s2[j] = Similarity(V[Sample_Not[j][0]], V[Sample_Not[j][1]])
    #比较优化之后的结果 小大比较 实现最小值大于最大值 以减少比较次数
    s1 = sorted(s1)
    print(len(s1))
    s2 = sorted(s2,reverse=True)
    print(len(s2))
    n1 = 0
    n2 = 0
    n3 = 0
    count_times = 0
    for i1 in range(0, len(s1)):
        t1 = 0
        t2 = 0
        for i2 in range(0, len(s2)):
            if s1[i1] < s2[i2]:
                n1 += 1
                t1 += 1
                count_times +=1
            elif s1[i1] == s2[i2]:
                n2 += 1
                t2 += 1
                count_times += 1
            else:
                add = len(s2)-t1-t2
                n3 += add
                break
    auc = (n3 + 0.5 * n2) / n
    print(count_times)
    return auc

E = np.loadtxt('D:/deepwalk_test/Jazz/Jazz_standard.txt',dtype=int)
Train = np.loadtxt('D:/deepwalk_test/Jazz/Train.edgelist',dtype=int)
Test = np.loadtxt('D:/deepwalk_test/Jazz/Test.edgelist',dtype=int)
V = np.loadtxt('D:/deepwalk_test/Jazz/Jazz_vector.txt',dtype=float,skiprows=1)
V = V[np.lexsort(V[:,::-1].T)]
V = np.delete(V,0,axis=1)

N = [[1]*len(V) for i in range(len(V))]
N = np.array(N)
#将真实边置零
for i in range(0,len(E)):
    a = E[i][0]
    b = E[i][1]
    N[a][b] = 0
#将自连接边置零
for i in range(0,len(V)):
    N[i][i] = 0
count2 =0
for i in range(0,len(V)):
    for j in range(0,len(V)):
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
for i in range(0,len(V)):
    for j in range(0, len(V)):
        if N[i][j] == 1:
            NotA[count3] = i
            NotB[count3] = j
            count3 += 1
Not = np.vstack([NotA,NotB])
Not=np.transpose(Not)

auc =AUC(1)
print(auc)