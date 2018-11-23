# coding=gbk
import numpy as np
import random

def CN(MatrixAdjacency_Train):
    Matrix_similarity = np.dot(MatrixAdjacency_Train, MatrixAdjacency_Train)
    return Matrix_similarity

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
    for i in range(len(Test)):
        s1[i] = S_CN[Test[i][0]][Test[i][1]]
    for i in range(len(Not)):
        s2[i] = S_CN[Not[i][0]][Not[i][1]]
    s1 = sorted(s1)
    print(s1)
    print(len(s1))
    np.savetxt('D:/deepwalk_test/USAir/M_s1.txt', s1, fmt="%d")
    s2 = sorted(s2,reverse=True)
    print(len(s2))
    np.savetxt('D:/deepwalk_test/USAir/M_s2.txt', s2, fmt="%d")
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

E = np.loadtxt('D:/deepwalk_test/USAir/USAir_standard.txt',dtype=int)
Train = np.loadtxt('D:/deepwalk_test/USAir/Train.edgelist',dtype=int)
Test = np.loadtxt('D:/deepwalk_test/USAir/Test.edgelist',dtype=int)
P_E = 0
for i in range(len(E)):
    if E[i][0] > P_E:
        P_E = E[i][0]
    elif E[i][1] > P_E:
        P_E = E[i][1]
P_E += 1
#构造Train的邻接矩阵
Train_AD = [[0]*P_E for i in range(P_E)]
for i in range(len(Train)):
    Train_AD[Train[i][0]][Train[i][1]] = 1

'''
list = [[0]for i in range(P_E)]
list = np.array(list)
for i in range(P_E):
    for j in range(P_E):
        if Train_AD[i][j] == 1 :
            list[i] +=1

print(list)
count4 = 0
for i in range(len(list)):
    count4 += list[i]
print(count4)
'''
S_CN = CN(Train_AD)
count = 0
for i in range(P_E):
    for j in range(P_E):
        if Train_AD[i][j] == 1:
            count +=1

'''
Not_M = [[1]*P_E for i in range(P_E)]
Not_M = np.array(Not_M)
for i in range(P_E):
    Not_M[i][i] = 0
for i in range(len(E)):
    Not_M[E[i][0]][E[i][1]] = 0
Num_Not_edge = P_E*P_E-P_E-len(E)
Not = [[0,0] for i in range(Num_Not_edge)]
Not = np.array(Not)
count = 0
for i in range(P_E):
    for j in range(P_E):
        if Not_M[i][j] == 1:
            Not[count][0]=i
            Not[count][1]=j
            count += 1
'''
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
