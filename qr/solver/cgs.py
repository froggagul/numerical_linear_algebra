import numpy as np
from .solver import SolverFactory, QRSolver

@SolverFactory.register('cgs')
class CGSQRSolver(QRSolver):
    def solve(self, a: np.ndarray):
        assert len(a.shape) == 2
        m, n = a.shape
        q = np.zeros(shape=(m, n))
        r = np.zeros(shape=(n, n))

        for j in range(n):
            v_j = a[:, j]
            for i in range(0, j):
                r[i][j] = q[:, i]@a[:, j]
                v_j = v_j - r[i][j] * q[:, i]
            r[j][j] = np.linalg.norm(v_j)
            q[:, j] = v_j / r[j][j]

        return q, r


if __name__ == "__main__":
    solver = SolverFactory.create_solver('cgs')

    a = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
    # a = np.array([[2, 3, 3], [3, 2, 2], [1, 1, 5]])
    q, r = solver.solve(a)
    print(q)
    print(r)
    print(a)
    print(q@r)
