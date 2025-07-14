import numpy as np

from accuracy_bounds.algo.projections import projection_nullspace_operator

def compute_feasible_set(A, input_data_point, target_data, q, epsilon):
    """
    Implements the iterative algorithm for estimating feasible sets for one input point based on possible target data points for a noisy inverse problem. 
        Arguments:
        - A: The matrix (for which we are computing the Moore-Penrose inverse) of the inverse problem input_data = A(target_data)+noise.
        - input_data_point: Input data point, referred to as "y" in variable names, for an approximate inverse method.
        - target_data: Target or ground truth data for an approximate inverse method.
        - q: order of the norm on the input dataset, default q=2 for MSE computation.
        - epsilon: Noise level in the inverse problem input_data = A(target_data)+noise.

        Returns:
        - feas_set_approx: approximation to the feasible set, consisting of all target data points that can map to an input data point within the noise level.
    """

    if isinstance(target_data, np.ndarray) and target_data.ndim==2: 
        # set batch size to 1
        target_data = target_data.reshape([target_data.shape[0], 1, target_data.shape[1]])
        
    feas_list_y = []

    for target_data_batch in target_data:
        e_n = input_data_point - np.dot(target_data_batch, A.T) # Compute noise vector

        # Check if noise is below noiselevel
        feas_mask_y = np.linalg.norm(e_n,q,axis=-1) <= epsilon
        
        feas_list_y.append(target_data_batch[feas_mask_y])

    return np.concatenate(feas_list_y, 0)

def diams_feasibleset_inv_sym(A, input_data_point, target_data, p,q, epsilon):
    """
    Implements the iterative algorithm for diameter estimation of the feasible set for a noisy inverse problem under the assumption that 
    there exists noise vectors such that the target data is symmetric with respect to the null space of F= (A |I): P_{N(F)^perp}(x,e)-P_{N(F)}(x,e). 
    Computes diameter based on possible target data points for one input point.
        Arguments:
        - A: The matrix (for which we are computing the Moore-Penrose inverse) of the inverse problem input_data = A(target_data)+noise.
        - input_data_point: Input data point, referred to as "y" in variable names, for an approximate inverse method.
        - target_data: Target or ground truth data for an approximate inverse method.
        - p: order of the norm on the target dataset, default p=2 for MSE computation.
        - q: order of norm on the input dataset.
        - epsilon: Noise level in the inverse problem input_data = A(target_data)+noise.

        Returns:
        - diameter_mean_y, num_feas, max_diam_Fy: diameter_mean_y of dim(0)= shape(input_data), the estimated mean diameter of the feasible set, 
                                        consisting of all possible target data points, for one input point.
                                        num_feas is the number of samples in the feasible set and will be used for statistics later on.
                                        max_diam_Fy the maximum diameter of the feasible set, 
                                        consisting of all possible target data points, for one input point.
    """
    # Compute Moore-Penrose-Inverse of F
    F = np.hstack((A, np.eye(A.shape[0])))  # Construct F: (A | I) 
    proj_ns_F = projection_nullspace_operator(F)

    # Step 2: Compute feasible sets and diameters
    max_diam_Fy = 0
    diameter_mean_y = 0
    diam_y = []

    for x_n in target_data:
        xcomp = len(x_n)-1
        e_n = input_data_point - np.dot(A,x_n) # Compute noise vector

        if np.linalg.norm(e_n,q) <= epsilon:  # Check if noise is below noiselevel
            # Project onto the null space of F
            proj_nullspace = np.dot(proj_ns_F, np.hstack((x_n, e_n)))[0:xcomp] # Project (x_n, e_n) onto nullspace of F, only take dim of x_n

            # Compute diameter of feasible set based on projection onto null space
            diameter = 2 * np.linalg.norm(proj_nullspace, ord = p)   

            #add to diam_y 
            diam_y.append(diameter)

            #get ascending diams
            if diameter > max_diam_Fy:
                max_diam_Fy = diameter
                
    # obtain number of samples in feasible set (num_feas will be used for statistics later on)
    num_feas = len(diam_y)        
    # get mean over diams 
    if num_feas > 0:      
        diameter_mean_y = np.mean(np.power(diam_y,p))

    return diameter_mean_y, num_feas, max_diam_Fy


def diams_feasibleset_inv(A, input_data_point, target_data, p, epsilon):
    """
    Implements the iterative algorithm for diameter estimation of the feasible set for a noisy inverse problem. 
    Computes diameter based on possible target data points for one input point.
        Arguments:
        - A: The matrix (for which we are computing the Moore-Penrose inverse) of the inverse problem input_data = A(target_data)+noise.
        - input_data_point: Input data point, referred to as "y" in variable names, for an approximate inverse method.
        - target_data: Target or ground truth data for an approximate inverse method.
        - p: order of the norm on the target dataset, default p=2 for MSE computation.
        - epsilon: Noise level in the inverse problem input_data = A(target_data)+noise.

        Returns:
        - diameter_mean_y, num_feas, max_diam_Fy: diameter_mean_y of dim(0)= shape(input_data), the estimated mean diameter of the feasible set, 
                                        consisting of all possible target data points, for one input point.
                                        num_feas is the number of samples in the feasible set and will be used for statistics later on.
                                        max_diam_Fy the maximum diameter of the feasible set, 
                                        consisting of all possible target data points, for one input point.
    """

    # Compute feasible sets and diameters
    feas_set_y = compute_feasible_set(A, input_data_point, target_data, p, epsilon)
        
    dist_ns = (feas_set_y[:,None,:] - feas_set_y[None,:,:])
    
    max_diam_Fy = -1
    diameter_mean_y = -1
    diam_y = []

    # obtain number of samples in feasible set (num_feas will be used for statistics later on)
    num_feas = len(feas_set_y)
    
    if num_feas == 0: 
        diameter_mean_y = 0
        max_diam_Fy = 0

    else:
        diameter = np.linalg.norm(dist_ns, ord = p, axis=-1)
        mask = np.tril(np.ones((num_feas, num_feas), dtype=bool)) # include diagonal elements (is always 0)

        diam_y = diameter[mask]
        max_diam_Fy = diam_y.max()

        # compute 2 times sum over diams to the power p divided by num_feas^2
        diameter_mean_y = 2*np.divide(np.sum(np.power(diam_y,p)), np.power(num_feas,2))

    return diameter_mean_y, num_feas, max_diam_Fy