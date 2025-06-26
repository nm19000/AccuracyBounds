import numpy as np

def projection_kernel(A, x):
    """
        Computes the projection of a point x onto the kernel or null space of A, i.e., P_{\mathcal{N}(A)}(x).
        This is equivalent to (I - A^dagger A) x.
    """
    A_dagger = np.linalg.pinv(A)
    A_daggerA = np.dot(A_dagger, A)
    projection = np.eye(A.shape[1]) - A_daggerA
    return np.dot(projection, x)

