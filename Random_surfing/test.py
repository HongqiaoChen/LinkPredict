

def Standard(matrix):
    out = [[0.0 for i in range(Num)]for j in range(Num)]
    out = np.array(out)
    for i in range(Num):
        out[i] = matrix[i]/sum(matrix[i])
    return out