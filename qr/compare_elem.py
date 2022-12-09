import numpy as np
from cgs import classical_gram_schmidt
from mgs import modified_gram_schmidt
from householder import householder

import matplotlib.pyplot as plt
import seaborn as sns

def orthogonality_error_array(q: np.ndarray):
    return q.T @ q - np.eye(N=q.shape[0])

if __name__ == "__main__":
    
    # As = np.random.randint(low = -16, high = 16, size=(100, 3, 3))
    m = 10
    n = 10
    As = np.random.rand(1000, m, n)
    
    indexs = []
    xs = []

    errors = np.zeros(shape=(3, m, n))

    for A in As:
        if np.linalg.det(A) == 0: # A must be full rank
            continue
        for i, func in enumerate([classical_gram_schmidt, modified_gram_schmidt, householder]):
            q, _ = func(A)
            error = orthogonality_error_array(q)
            errors[i] += np.abs(error)
        errors /= As.shape[0]

    
    fig, axs = plt.subplots(1, 3, sharey=True)
    # fig.title('cgs vs mgs vs householder orthogonality')
    titles = ["cgs", "mgs", "householder"]
    for i, ax in enumerate(axs):
        ax.set_title(titles[i])
        sns.heatmap(errors[i], linewidth=0.5, ax = ax, vmin=0, vmax=8e-19)
 
    plt.show()

