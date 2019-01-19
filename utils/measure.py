import numpy as np
from numpy import linalg


def cosine_similarity(vector1, vector2):
    # if vector1 and vector2 are column vector, we can use:
    # vector1.T * vector2
    val = np.dot(vector1, vector2)
    # or:
    # val = vector1.dot(vector2.T)
    norm = linalg.norm(vector1) * linalg.norm(vector2)
    cos = val / norm
    sim = 0.5 + 0.5 * cos  # normalization
    return sim


def euclidean_metric(vector1, vector2):
    dist = linalg.norm(vector1 - vector2)
    met = 1.0 / (1.0 + dist)  # normalization
    return met


def pearson_correlation(vector1, vector2):
    x = vector1 - np.mean(vector1)
    y = vector2 - np.mean(vector2)
    d = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
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
    return (po - pe) / (1 - pe)
