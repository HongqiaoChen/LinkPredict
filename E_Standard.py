import numpy as np

E = np.loadtxt('/home/hongqiaochen/Desktop/Link_predict/Power/Power.txt')
#得到无权图
if len(E[0])==3 :
    E = np.delete(E,2,axis=1)
#得到从0开始编号的节点集构成的边集
E = np.transpose(E)
list1 = E[0]
list2 = E[1]
E = np.transpose(E)
list1 = list1.tolist()
list2 = list2.tolist()
list3 = list1+list2
list3 = np.unique(list3)
list3 = np.array(list3)
list3 = list3.reshape(-1,1)
if np.min(list3) != 0:
    dec = np.min(list3)
    dec_m = [[dec,dec] for i in range(len(E))]
    dec_m = np.array(dec_m)
    E = E-dec_m


# 将E的float类型改为int类型（原始数据集可能是一个有权图，其权重大概率为浮点数）
E = E.astype(dtype = np.int32)

# 删除重复边
count = 0
for i in range(len(E)):
    if E[i][0] > E[i][1]:
        temp = E[i][0]
        E[i][0] = E[i][1]
        E[i][1] = temp
        count += 1
l = len(np.unique(E))
list4 = [[0]*l for i in range(l)]
list4 = np.array(list4)
for i in range(len(E)):
    list4[E[i][0]][E[i][1]] +=1
count = 0
for i in range(l):
    for j in range(l):
        if list4[i][j] > 0 :
            count += 1
list5 = [[0,0] for i in range(count)]
list5 =np.array(list5)

count = 0
for i in range(l):
    for j in range(l):
        if list4[i][j] > 0:
            list5[count][0] = i
            list5[count][1] = j
            count += 1
E = list5

e = np.unique(E)
if e[-1]+1 == len(e):
    print('OK')
    np.savetxt('/home/hongqiaochen/Desktop/Link_predict/Power/Power_standard.txt', E, fmt="%d %d")
else:
    print('Worng')
