import numpy as np
from .utils import projection_nullspace, apply_forwardmodel


def diams_feasibleset_inv(A, input_data_point, target_data, p=2, epsilon=1e-1):
    """
    Implements the iterative algorithm for diameter estimation of the feasible set for a noisy inverse problem, consisting of all possible target data points, for one input point.
        Arguments:
        - A: The matrix (for which we are computing the Moore-Penrose inverse) of the inverse problem input_data = A(target_data)+noise.
        - input_data_point: Input data point, referred to as "y" in variable names, for an approximate inverse method.
        - target_data: Target or ground truth data for an approximate inverse method.
        - p: order of the norm, default p=2 for MSE computation.
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

    # Step 2: Compute diameters
    max_diam_Fy = 0
    diameter_mean_y = 0
    diam_y = []

    for x_n in target_data:
        xcomp = len(x_n)
        e_n = input_data_point - np.dot(A,x_n) # Compute noise vector

        if np.linalg.norm(e_n,p) <= epsilon:  # Check if noise is below noiselevel
            # Project onto the null space of F
            proj_nullspace = projection_nullspace(F, np.hstack((x_n, e_n)))[0:xcomp] # Project (x_n, e_n) onto nullspace of F, only take dim of x_n

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


def wc_kernelsize(A, input_data, target_data, p, epsilon=1e-1):
    """
    Computes the worst-case kernel size for noisy inverse problem using Algorithm 1.

    Args:
        - A: The matrix (for which we are computing the Moore-Penrose inverse) of the inverse problem input_data = A(target_data)+noise.
        - input_data: Input data for an approximate inverse method.
        - target_data: Target or ground truth data for an approximate inverse method.
        - p: order of the norm, default p=2 for MSE computation.
        - epsilon: Noise level in the inverse problem input_data = A(target_data)+noise.

    Returns:
        Approximate worst-case kernel size for a set of input data samples.
    """
    wc_kersize =0

    for y in input_data:
        # compute diameter of feasible set for one input data point
        diameter_mean_y, num_feas, max_diam_Fy = diams_feasibleset_inv(A, y, target_data, p, epsilon)
        if max_diam_Fy > wc_kersize:
            wc_kersize = max_diam_Fy
    
    return wc_kersize


def av_kernelsize(A, input_data, target_data, p, epsilon=1e-1):
    """
    Computes the average kernel size for a noisy inverse problem under Algorithm 2.

    Args:
        - A: The matrix (for which we are computing the Moore-Penrose inverse) of the inverse problem input_data = A(target_data)+noise.
        - input_data: Input data for an approximate inverse method.
        - target_data: Target or ground truth data for an approximate inverse method.
        - p: order of the norm, default p=2 for MSE computation.
        - epsilon: Noise level in the inverse problem input_data = A(target_data)+noise.

    Returns:
        Approximate average kernel size for for a set of input data samples.
    """

    av_kersizep = 0
    num_samples = len(input_data)

    for y in input_data:
        # compute diameter of feasible set for one input data point (num_feas will be used for statistics later on)
        diameter_mean_y, num_feas, max_diam_Fy = diams_feasibleset_inv(A, y, target_data, p, epsilon)
        #add diameters means for obtaining average kersize to the power p
        av_kersizep = av_kersizep + diameter_mean_y
        
    # get mean over input data
    av_kersize = np.divide(av_kersizep, num_samples)    
    # take power 1/p to obtain average kersize
    av_kersize =  np.power(av_kersize, 1/p)
    
    return av_kersize
