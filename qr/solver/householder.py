import numpy as np
from typing import Union
from .solver import QRSolver, SolverFactory

@SolverFactory.register('householder')
class HouseHolderQRSolver(QRSolver):
    def solve(self, a: np.ndarray):
        assert len(a.shape) == 2
        m, n = a.shape
        assert m >= n
    
        q = np.eye(N=m)
        r = a.copy()
    
        for k in range(n):
            x = r[k:m, k]
            v_k = x.copy()
            v_k[0] += np.sign(x[0]) * np.linalg.norm(x)
            v_k = v_k / np.linalg.norm(v_k)
            # sub = (v_k @ r[k:m, k:n]).reshape(1, -1)
            # sub = 2 * v_k.reshape(-1, 1) @ sub
            # r[k:m, k:n] = r[k:m, k:n] - sub
            
            q_i = np.eye(N=m)
            q_i[k:, k:] -= 2 * v_k.reshape(-1, 1) @ v_k.reshape(1, -1)
            
            q = q @ q_i # q = q_1 q_2 q_3 ...
            r = q_i @ r
    
        return q, r

if __name__ == "__main__":
    solver = SolverFactory.create_solver('householder')
    a = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]], dtype=np.float64)
    # a = np.array([[2, 3, 3], [3, 2, 2], [1, 1, 5]])
    # a = np.random.rand(5, 4)
    # q, r = householder_qr_decomposition(a)
    q, r = solver.solve(a)

    print(q)
    print(r)
    print(a)
    print(q@r)
