import numpy as np

def modified_gram_schmidt(a: np.ndarray):
    assert len(a.shape) == 2
    m, n = a.shape
    q = np.zeros(shape=(m, n))
    r = np.zeros(shape=(n, n))

    v = np.zeros(shape=(m, n))
    for i in range(n):
        v[:, i] = a[:, i]

    for i in range(n):
        r[i][i] = np.linalg.norm(v[:, i])
        q[:, i] = v[:, i] / r[i][i]
        for j in range(i + 1, n):
            r[i][j] = np.dot(q[:, i].T, v[:, j])
            v[:, j] = v[:, j] - r[i][j] * q[:, i]
            
    return q, r

if __name__ == "__main__":
    a = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
    # a = np.array([[1, 0], [0, 1]])
    q, r = modified_gram_schmidt(a)
    print(q)
    print(r)
    print(a)
    print(np.matmul(q, r))
    print(np.allclose(a, np.matmul(q, r)))

