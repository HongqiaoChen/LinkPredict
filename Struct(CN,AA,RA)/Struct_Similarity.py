import numpy as np

def get_sample(Test,Not):
    l_test = len(Test)
    l_Not = len(Not)
    MAX = 672400
    Test_sample = np.random.choice(l_test, size=MAX, replace=True)
    Not_sample = np.random.choice(l_Not, size=MAX, replace=True)
    return Test_sample,Not_sample

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

def AUC(Test_sample,Not_sample,f):
    MAX = 672400
    S_Test_Sample = [0 for i in range(MAX)]
    S_Not_Sample = [0 for i in range(MAX)]
    for i in range(MAX):
        S_Test_Sample[i] = f(Test[Test_sample[i]][0],Test[Test_sample[i]][1])
    for i in range(MAX):
        S_Not_Sample[i] = f(Not[Not_sample[i]][0],Not[Not_sample[i]][1])
    n = MAX
    n1 = 0
    n2 = 0
    for i in range(MAX):
        if S_Test_Sample[i] > S_Not_Sample[i]:
            n1 += 1
        if S_Test_Sample[i] == S_Not_Sample[i]:
            n2 += 1
    auc = (n1 + 0.5 * n2) / n
    return auc


# 获取Test，Train，E集
Test = np.loadtxt('/home/hongqiaochen/Desktop/Link_predict/USAir/Test.edgelist',dtype=int)
E = np.loadtxt('/home/hongqiaochen/Desktop/Link_predict/USAir/USAir_standard.txt',dtype=int)
Train = np.loadtxt('/home/hongqiaochen/Desktop/Link_predict/USAir/Train.edgelist',dtype=int)


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

# 取样

Test_sample,Not_sample = get_sample(Test,Not)
RA = AUC(Test_sample,Not_sample,RA_Similarity)
print(RA)
AA = AUC(Test_sample,Not_sample,AA_Similarity)
print(AA)
CN = AUC(Test_sample,Not_sample,CN_Similarity)
print(CN)
