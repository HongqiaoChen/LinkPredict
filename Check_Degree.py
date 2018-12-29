import numpy as np

path = '/home/hongqiaochen/Desktop/Date_Link_predict/WS'
train = np.loadtxt(path+'/Train.edgelist',dtype=int)

A = np.unique(train)[-1]+1
B = len(np.unique(train))

if A == B:
    print('OK')
else:
    print('Wrong')
