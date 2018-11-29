import numpy as np

param_p = np.arange(0,1.1,0.5)
print(len(param_p))
param_times = np.arange(5,20,5)
print(len(param_times))
param_length = np.arange(10,50,10)
print(len(param_length))
param_window = np.arange(5,50,5)
print(len(param_window))
param_min_count = np.array([0])
print(len(param_min_count))
param_alpha = np.array([0.001,0.01,0.025,0.05,0.1,0.25,0.5,1])
print(len(param_alpha))

# AUC = np.loadtxt('/home/hongqiaochen/Desktop/Link_predict/Yeast/bestDW.edgelist',dtype=float)
# AUC = AUC.tolist()
# best_auc_index = AUC.index(max(AUC))
# best_auc = AUC[best_auc_index]
# print(best_auc)
best_auc_index = 57
n1 = len(param_times)*len(param_length)*len(param_window)*len(param_min_count)*len(param_alpha)
n2 = len(param_length)*len(param_window)*len(param_min_count)*len(param_alpha)
n3 = len(param_window)*len(param_min_count)*len(param_alpha)
n4 = len(param_min_count)*len(param_alpha)
n5 = len(param_alpha)
c1 = best_auc_index//n1
m1 = best_auc_index%n1
c2 = m1//n2
m2 = m1%n2
c3 = m2//n3
m3 = m2%n3
c4 = m3//n4
m4 = m3%n4
c5 = m4//n5
m5 = m4%n5
c6 = m5
print(param_p[c1],param_times[c2],param_length[c3],param_window[c4],param_min_count[c5],param_alpha[c6])
