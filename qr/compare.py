import numpy as np
from cgs import classical_gram_schmidt
from mgs import modified_gram_schmidt
from householder import householder

import matplotlib.pyplot as plt

def orthogonality_error(q: np.ndarray):
    return np.linalg.norm(q.T @ q - np.eye(N=q.shape[0]))

if __name__ == "__main__":
    
    # As = np.random.randint(low = -16, high = 16, size=(100, 3, 3))
    As = np.random.rand(10000, 4, 4)
    cgs_errors = []
    mgs_errors = []
    householder_errors =[]
    
    indexs = []
    xs = []

    for i, A in enumerate(As):
        if np.linalg.det(A) == 0: # A must be full rank
            continue

        q, _ = classical_gram_schmidt(A)
        cgs_error = orthogonality_error(q)
        cgs_errors.append(cgs_error)

        q, _ = modified_gram_schmidt(A)
        mgs_error = orthogonality_error(q)
        mgs_errors.append(mgs_error)
        
        q, _ = householder(A)
        householder_error = orthogonality_error(q)
        householder_errors.append(householder_error)
        
        # xs.append(np.linalg.det(A))
        # xs.append(np.linalg.cond(A))
        xs.append(i)
    #        if mgs_error > cgs_error:
    #            print('mgs > cgs\n', A, np.linalg.det(A))
    #        if householder_error > mgs_error:
    #            print('house > mgs\n', A, np.linalg.det(A))

    plt.scatter(xs, cgs_errors, label='cgs')
    plt.scatter(xs, mgs_errors, label='mgs')
    plt.scatter(xs, householder_errors, label='householder')

    plt.legend()
    plt.ylabel('orthogonality error')
    plt.xlabel('condition number')
    plt.title('cgs vs mgs vs householder orthogonality')
    
    plt.show()

