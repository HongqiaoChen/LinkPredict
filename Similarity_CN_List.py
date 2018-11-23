import numpy as np
import random

def Similarity(V1,V2):
    list_s =list(set(List1[V1]).intersection(set(List2[V2])))
    return len(list_s)

def AUC(rate):
    Number_Sample_Test = int(len(Test) * rate)  # 伯努利分布，取样去估计总体。现在样本为总体的
    print(Number_Sample_Test)
    Number_Sample_Not = int(len(Not) * rate)  # 伯努利分布，取样去估计总体。现在样本为总体的
    print(Number_Sample_Not)
    Sample_Test = [[0, 0] for i in range(Number_Sample_Test)]
    Sample_Not = [[0, 0] for i in range(Number_Sample_Not)]
    r1 = random.sample(range(0, len(Test)), Number_Sample_Test);
    r2 = random.sample(range(0, len(Not)), Number_Sample_Not);
    for i in range(0, Number_Sample_Test):
        Sample_Test[i] = Test[r1[i]]
    for i in range(0, Number_Sample_Not):
        Sample_Not[i] = Not[r2[i]]
    n = len(Sample_Test) * len(Sample_Not)
    s1 = [[0] for i in range(Number_Sample_Test)]
    s1 = np.array(s1)
    s2 = [[0] for i in range(Number_Sample_Not)]
    s2 = np.array(s2)
    for i in range(len(Sample_Test)):
        s1[i] = Similarity(Sample_Test[i][0],Sample_Test[i][1])
    for j in range(len(Sample_Not)):
        s2[j] = Similarity(Sample_Not[j][0], Sample_Not[j][1])
    #排序减少比较次数
    s1 = sorted(s1)
    print(s1)
    s2 = sorted(s2,reverse=True)
    print(s2)
    n1 = 0
    n2 = 0
    n3 = 0
    for i1 in range(0, len(s1)):
        t1 = 0
        t2 = 0
        for i2 in range(0, len(s2)):
            if s1[i1] < s2[i2]:
                n1 += 1
                t1 += 1

            elif s1[i1] == s2[i2]:
                n2 += 1
                t2 += 1

            else:
                add = len(s2)-t1-t2
                n3 += add
                break
    auc = (n3 + 0.5 * n2) / n
    return auc

Test = np.loadtxt('D:/deepwalk_test/Jazz/Test.edgelist',dtype=int)
E = np.loadtxt('D:/deepwalk_test/Jazz/Jazz_standard.txt',dtype=int)
Train = np.loadtxt('D:/deepwalk_test/Jazz/Train.edgelist',dtype=int)
#创建List集 :List1和List2_代替邻接矩阵
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

#创建Not集
P_E = 0
for i in range(len(E)):
    if E[i][0] > P_E:
        P_E = E[i][0]
    elif E[i][1] > P_E:
        P_E = E[i][1]
P_E += 1
N = [[1]*P_E for i in range(P_E)]
N = np.array(N)
#将真实边置零
for i in range(0,len(E)):
    a = E[i][0]
    b = E[i][1]
    N[a][b] = 0
#将自连接边置零
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

auc = AUC(1)
print(auc)


