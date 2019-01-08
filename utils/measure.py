import numpy as np
from numpy import linalg


def cosine_similarity(vector1, vector2):
    # val = vector1.dot(vector2.T)  # 若为列向量则 vector1.T * vector2
    val = np.dot(vector1, vector2)
    norm = linalg.norm(vector1) * linalg.norm(vector2)
    cos = val / norm  # 余弦值
    sim = 0.5 + 0.5 * cos  # 归一化
    return sim


def euclidean_metric(vector1, vector2):
    dist = linalg.norm(vector1 - vector2)
    met = 1.0 / (1.0 + dist)  # 归一化
    return met


def pearson_correlation(vector1, vector2):
    x_ = vector1 - np.mean(vector1)
    y_ = vector2 - np.mean(vector2)
    d = np.dot(x_, y_) / (np.linalg.norm(x_) * np.linalg.norm(y_))
    return d


def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    print(po, pe)
    return (po - pe) / (1 - pe)
