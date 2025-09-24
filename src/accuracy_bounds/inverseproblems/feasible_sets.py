import numpy as np


def compute_feasible_set_linear_forwardmodel(A, input_data_point, target_data, p_2, epsilon):
    """
    Implements the iterative algorithm for estimating feasible sets for one input point based on possible target data points for a noisy inverse problem. 
    
    Args:
        - A: The matrix (for which we are computing the Moore-Penrose inverse) of the inverse problem input_data = A(target_data)+noise.
        - input_data_point: Input data point, referred to as "y" in variable names, for an approximate inverse method.
        - p_2: Order of the norm on the target dataset $\mathcal{M}_2 = A(\mathcal{M}_1)+\mathcal{E}$. Set to p=2 for the ell 2 norm computation.
        - epsilon: Noise level in the inverse problem input_data = A(target_data)+noise.

    Returns:
        - feas_set_y: approximation to the feasible set, consisting of all target data points that can map to an input data point within the noise level.
    """

    # Step 2: Compute feasible set for input data point y
    feas_set_y = []

    for x_n in target_data:
        e_n = input_data_point - np.dot(A,x_n) # Compute noise vector

        if np.linalg.norm(e_n,p_2) <= epsilon:  # Check if noise is below noiselevel
            # add traget data point x_n to feasible set
            feas_set_y.append(x_n)

    return feas_set_y
