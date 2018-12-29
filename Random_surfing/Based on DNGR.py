import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

path = '/home/hongqiaochen/Desktop/Date_Link_predict/Power'

max_step = 10
alpha = 0.98
goal_D = 100

def random_surfing_w(count):
    w = alpha**count+alpha**count*(1-alpha)*(max_step-count)
    return w

# def Initialization(shape):
#     init = tf.random_normal(shape,stddev=0.01)
#     return tf.Variable(init)
#
# def ML(x):
#     W1 = Initialization([Num,300])
#     W2 = Initialization([300,Num])
#     out = tf.matmul(tf.matmul(x,W1),W2)
#     return out

def get_sample(Test, Not):
    l_test = len(Test)
    l_Not = len(Not)
    MAX = 672400
    Test_sample = np.random.choice(l_test, size=MAX, replace=True)
    Not_sample = np.random.choice(l_Not, size=MAX, replace=True)
    return Test_sample, Not_sample

def DW_Similarity(V1,V2):
    temp = np.sqrt(np.sum(np.square(V1 - V2)))
    S = float(1/(1+temp))
    return S

def auc():
    MAX = 672400
    V = np.loadtxt(path+'/DNGR_vector.txt', dtype=float,skiprows=1)
    V = V[np.lexsort(V[:, ::-1].T)]
    V = np.delete(V, 0, axis=1)
    Test_sample, Not_sample = get_sample(Test, Not)
    S_Test_Sample = [0 for i in range(MAX)]
    S_Not_Sample = [0 for i in range(MAX)]
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
    auc = (n1+0.5*n2)/n
    return auc

E = np.loadtxt(path+'/standard.txt', dtype=int)
Num = len(np.unique(E))

adjlist = [[0 for i in range(Num)]for j in range(Num)]
adjarr = np.array(adjlist)

for i in range(len(E)):
    adjarr[E[i][0]][E[i][1]] = 1
    adjarr[E[i][1]][E[i][0]] = 1

Row_Sum = [0 for i in range(Num)]
for i in range(Num):
    row_sum = 0
    for j in range(Num):
        row_sum = row_sum + adjarr[i][j]
    Row_Sum[i] = row_sum

S = [[0.0 for i in range(Num)]for j in range(Num)]
S = np.array(S)
for i in range(Num):
    for j in range(Num):
        if Row_Sum[i] == 0:
            S[i][j] = 0
        else:
            S[i][j] = adjarr[i][j]/Row_Sum[i]

Y = [[0.0 for i in range(Num)]for j in range(Num)]
Y = np.array(Y)

P = [[0.0 for i in range(Num)]for j in range(Num)]
P = np.array(P)
for i in range(Num):
    P[i][i] = 1
P_init = P



for count in range(max_step):
    P = alpha * np.dot(P,S) + (1-alpha) * P_init
    Y = Y+random_surfing_w(count)*P

# for i in range(Num):
#     Y[i] = np.exp(Y[i])/sum(np.exp(Y[i]))

X = P_init
XI = np.linalg.inv(X)
W = np.dot(XI,Y)

# train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.2,random_state=234)
#
# x = tf.placeholder(tf.float32,[None,Num])
# y_ = tf.placeholder(tf.float32,[None,Num])
# y = ML(x)
#
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))
#
# # train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)
# train_step = tf.train.AdadeltaOptimizer(0.01).minimize(cross_entropy)
# # train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
#
# saver = tf.train.Saver()
# initial = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(initial)
#     for i in range(10000):
#         sess.run(train_step,{x:train_x,y_:train_y})
#         train_loss = sess.run(cross_entropy,{x:train_x,y_:train_y})
#         train_accuracy = sess.run(accuracy, {x:train_x, y_:train_y})
#         test_accuracy = sess.run(accuracy, {x:test_x, y_:test_y})
#         print("iter step %d, loss %f, training accuracy %f, test accuracy %f" % (i, train_loss, train_accuracy, test_accuracy))

U,sigma,VT = np.linalg.svd(W)

sigma_D = [[0.0 for i in range(goal_D)]for j in range(goal_D)]
sigma_D = np.array(sigma_D)
for i in range(goal_D):
    sigma_D[i][i] = sigma[i]
U_D = U[:,0:goal_D]

R = np.dot(U_D,sigma_D)

l = list(range(Num))
l = np.array(l)
R = np.insert(R,0,values=l,axis=1)
np.savetxt(path+'/DNGR_vector.txt',R,fmt='%f',delimiter=' ')
with open(path+'/DNGR_vector.txt', 'r+') as f:
    content = f.read()
    f.seek(0, 0)
    f.write('%d %d\n' % (Num, goal_D)+content)

# 读取Test,E,Train集合
Test = np.loadtxt(path+'/Test.edgelist',dtype=int)
E = np.loadtxt(path+'/standard.txt',dtype=int)
Train = np.loadtxt(path+'/Train.edgelist',dtype=int)
# 构造Not集
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

auc = auc()

print(auc)