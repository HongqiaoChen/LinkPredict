import numpy as np

E = np.loadtxt('D:/deepwalk_test/USAir/USAir_standard.txt',dtype=int)
list= [1265, 1855, 1293, 899, 1660, 1371, 275, 779, 1933, 813, 1247, 1144, 1349, 1490, 1935, 274, 1578, 822, 1380, 1959, 357, 885, 991, 1901, 1495, 1587, 1904, 1555, 291, 1693, 368, 567, 1362, 411, 339, 663, 69, 188, 669, 1605, 890, 1625, 917, 611, 228, 1364, 1620, 258, 1206, 607, 1506, 589, 2062, 1745, 1512, 1705, 844, 757, 867, 1266, 198, 2031, 113, 1091, 1936, 1944, 81, 1392, 1366, 1721, 1822, 951, 1445, 1186, 854, 1122, 1520, 629, 739, 279, 94, 732, 143, 1840, 1139, 182, 1695, 1594, 1849, 469, 166, 1147, 806, 165, 348, 130, 1152, 1981, 740, 1813, 266, 952,806]
Train = np.delete(E, list, axis=0)
np.savetxt('D:/deepwalk_test/test/Train.edgelist',Train,fmt="%d %d")