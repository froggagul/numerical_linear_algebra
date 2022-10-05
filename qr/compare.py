import numpy as np
from cgs import classical_gram_schmidt
from mgs import modified_gram_schmidt
from householder import householder

import matplotlib.pyplot as plt

def orthogonality_error(q: np.ndarray):
    return np.linalg.norm(q.T @ q - np.eye(N=q.shape[0]))

if __name__ == "__main__":
    As = np.random.rand(100, 3, 3)
    cgs_errors = []
    mgs_errors = []
    householder_errors =[]
    for A in As:
        q, _ = classical_gram_schmidt(A)
        cgs_errors.append(orthogonality_error(q))
        q, _ = modified_gram_schmidt(A)
        mgs_errors.append(orthogonality_error(q))
        
        q, _ = householder(A)
        householder_errors.append(orthogonality_error(q))

    plt.plot(cgs_errors, label='cgs')
    plt.plot(mgs_errors, label='mgs')
    plt.plot(householder_errors, label='householder')

    plt.legend()
    plt.ylabel('orthogonality error')
    plt.title('cgs vs mgs vs householder orthogonality')
    plt.savefig('compare_result.png')

