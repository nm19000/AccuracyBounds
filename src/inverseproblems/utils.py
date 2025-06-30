import numpy as np

# Function to apply matrix transformation A to points
def apply_forwardmodel(A, points):
    return np.dot(points, A.T)

def projection_nullspace(A, x):
    """
        Compute the projection of a point x onto the null space of A, i.e., P_{\mathcal{N}(A)}(x).
        This is equivalent to (I - A^dagger A) x.
    """
    A_dagger = np.linalg.pinv(A)
    return np.dot(np.eye(A.shape[1]) - np.dot(A_dagger, A), x)