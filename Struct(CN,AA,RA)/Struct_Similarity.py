import numpy as np

path = '/home/hongqiaochen/Desktop/Date_Link_predict/WS'
sigma = 1e-8
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
        S = 1/(np.log(len(l[CN[i]]))+sigma) + S
    return S

def Create_Not():
    Num = len(np.unique(E))
    N = [[1] * Num for i in range(Num)]
    N = np.array(N)
    for i in range(0, len(E)):
        N[E[i][0]][E[i][1]] = 0
        N[E[i][1]][E[i][0]] = 0
    for i in range(0, Num):
        N[i][i] = 0

    count = 0
    for i in range(0, Num):
        for j in range(0, Num):
            if N[i][j] == 1:
                count += 1
    number = int(count / 2)

    count = 0
    Not = [[0, 0] for i in range(number)]
    for i in range(Num):
        for j in range(i, Num):
            if N[i][j] == 1:
                Not[count][0] = i
                Not[count][1] = j
                count += 1
    return Not

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
Test = np.loadtxt(path+'/Test.edgelist',dtype=int)
E = np.loadtxt(path+'/standard.txt',dtype=int)
Train = np.loadtxt(path+'/Train.edgelist',dtype=int)

Not = Create_Not()

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
