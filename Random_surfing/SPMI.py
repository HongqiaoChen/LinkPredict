import numpy as np

path = '/home/hongqiaochen/Desktop/Date_Link_predict/WS'

max_step = 10
goal_D = 100

def Standard(matrix):
    out = [[0.0 for i in range(Num)]for j in range(Num)]
    out = np.array(out)
    for i in range(Num):
        out[i] = matrix[i]/sum(matrix[i])
    return out

def get_sample(Test, Not):
    # 获得Test集和Not集的长度　
    l_test = len(Test)
    l_Not = len(Not)
    MAX = 672400
    # 从中随机抽样672400次，得到Test、Not样本集（下标）.得到672400个下标
    Test_sample = np.random.choice(l_test, size=MAX, replace=True)
    Not_sample = np.random.choice(l_Not, size=MAX, replace=True)
    return Test_sample, Not_sample

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

# 二阶欧式距离
def DW_Similarity(V1,V2):
    temp = np.sqrt(np.sum(np.square(V1 - V2)))
    S = float(1/(1+temp))
    return S

def AUC(Test,Not):
    MAX = 672400
    V = np.loadtxt(path+'/Katz_vector.txt', dtype=float,skiprows=1)
    V = V[np.lexsort(V[:, ::-1].T)]
    V = np.delete(V, 0, axis=1)
    Test_sample, Not_sample = get_sample(Test, Not)

    S_Test_Sample = [0 for i in range(MAX)]
    S_Not_Sample = [0 for i in range(MAX)]
    # S_Test_Sample为Test_Sample所选取下标（672400）对应的Test中边的相似性（具体落脚点在边端点的相似性）
    for i in range(MAX):
        S_Test_Sample[i] = DW_Similarity(V[Test[Test_sample[i]][0]], V[Test[Test_sample[i]][1]])
    for j in range(MAX):
        S_Not_Sample[j] = DW_Similarity(V[Not[Not_sample[j]][0]], V[Not[Not_sample[j]][1]])
    n = MAX
    n1 = 0
    n2 = 0
    for i in range(MAX):
        if S_Test_Sample[i] > S_Not_Sample[i]:
            n1 += 1
        if S_Test_Sample[i] == S_Not_Sample[i]:
            n2 += 1
    AUC = (n1+0.5*n2)/n
    return AUC


# 读取Test,E,Train集合
E = np.loadtxt(path+'/standard.txt',dtype=int)
Test = np.loadtxt(path+'/Test.edgelist',dtype=int)
Train = np.loadtxt(path+'/Train.edgelist',dtype=int)
# 构造Not集
Not = Create_Not()

Num = len(np.unique(E))
adjlist = [[0.0 for i in range(Num)]for j in range(Num)]
adjarr = np.array(adjlist)
for i in range(len(E)):
    adjarr[E[i][0]][E[i][1]] = 1
    adjarr[E[i][1]][E[i][0]] = 1

P = [[0.0 for i in range(Num)]for j in range(Num)]
P = np.array(P)
for i in range(Num):
    P[i][i] = 1
P_start = P

Temp_adjarr = adjarr
Temp_standard = Standard(Temp_adjarr)
Y = Temp_standard

for count in range(max_step):
    Temp_adjarr = np.dot(Temp_adjarr,adjarr)
    Temp_standard = Standard(Temp_adjarr)
    Y = Y+Temp_standard



for i in range(Num):
    for j in range(Num):
        if Y[i][j] == 0 :
            Y[i][j] == 10e-8
        else:
            Y[i][j]= np.log10(Y[i][j])
print(Y)


U,sigma,VT = np.linalg.svd(Y)


sigma_D = [[0.0 for i in range(goal_D)]for j in range(goal_D)]
sigma_D = np.array(sigma_D)
for i in range(goal_D):
    sigma_D[i][i] = sigma[i]
U_D = U[:,0:goal_D]

R = np.dot(U_D,sigma_D)

l = list(range(Num))
l = np.array(l)
R = np.insert(R,0,values=l,axis=1)
np.savetxt(path+'/Katz_vector.txt',R)
with open(path+'/Katz_vector.txt', 'r+') as f:
    content = f.read()
    f.seek(0, 0)
    f.write('%d %d\n' % (Num, goal_D)+content)

AUC = AUC(Test,Not)

print(AUC)
