import numpy as np
import torch
from scipy.sparse import csr_matrix, lil_matrix
from joblib import Parallel, delayed
from .utils import projection_nullspace_operator

# plain python versions

def diams_feasibleset(feasible_set_y, p_1 ,p):
    """
    Implements computation of the diameter of the feasible set for a inverse problem. 
    Computes diameter based on a feasible set for one measurement or also input data point.
    
    Args:
        - feasible_set_y: feasible set for one measurement y.
        - p_1: Order of the norm on the target dataset $\mathcal{M}_1$. Set to p=2 for the ell 2 norm computation.
        - p: Order of the average kernel size. Set to p=2 for the MSE lower bound computation and p=1 for MAE lower bound computation.
    
    Returns:
        - diameter_mean_y, num_feas, max_diam_Fy: diameter_mean_y of dim(0)= shape(input_data), the estimated mean diameter of the feasible set to the power p, 
                                        consisting of all possible target data points, for one input point.
                                        num_feas is the number of samples in the feasible set and will be used for statistics later on.
                                        max_diam_Fy the maximum diameter of the feasible set, 
                                        consisting of all possible target data points, for one input point.
    """

    # Step 2: Compute feasible sets and diameters
    feas_set_y = feasible_set_y
    max_diam_Fy = 0
    diameter_mean_y = 0
    diam_y = []
    # obtain number of samples in feasible set (num_feas will be used for statistics later on)
    num_feas = len(feas_set_y)
    # compute diameters
    for h in range(0,num_feas,1):
        for j in range(0,h+1,1):
            #compute vectors in null space of F and their norm
            dist_ns = feas_set_y[h]-feas_set_y[j]
            diameter= np.linalg.norm(dist_ns, ord = p_1)

            #add to diam_y 
            diam_y.append(diameter)

            #get ascending diams
            if diameter > max_diam_Fy:
                max_diam_Fy = diameter
                            
    # get mean over diams, with factor 2 due to symmetry of the norm of the compute vectors in null space of F (norm(x-z)=norm(z-x))
    # and divided by num_feas^2 ad we have that many terms
    if num_feas > 0:      
        # compute 2 times sum over diams to the power p divided by num_feas^2
        diameter_mean_y = 2*np.divide(np.sum(np.power(diam_y,p)), np.power(num_feas,2))
    elif num_feas==0:
        diameter_mean_y = 0  

    return diameter_mean_y, num_feas, max_diam_Fy

def diams_feasibleset_linear_forwardmodel_sym(A, input_data_point, target_data, p_1, p_2, p, epsilon):
    """
    Implements the iterative algorithm for diameter estimation of the feasible set for a noisy inverse problem under the assumption that 
    there exists noise vectors such that the target data is symmetric with respect to the null space of F= (A |I): P_{N(F)^perp}(x,e)-P_{N(F)}(x,e). 
    Computes diameter based on possible target data points for one input point.
    
    Args:
        - A: The matrix (for which we are computing the Moore-Penrose inverse) of the inverse problem input_data = A(target_data)+noise.
        - input_data_point: Input data point, referred to as "y" in variable names, for an approximate inverse method.
        - target_data: Target or ground truth data for an approximate inverse method.
        - p_1: Order of the norm on the target dataset $\mathcal{M}_1$. Set to p=2 for the ell 2 norm computation.
        - p_2: Order of the norm on the target dataset $\mathcal{M}_2 = A(\mathcal{M}_1)+\mathcal{E}$. Set to p=2 for the ell 2 norm computation.
        - p: Order of the average kernel size. Set to p=2 for the MSE lower bound computation and p=1 for MAE lower bound computation.
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
    norm_projection_y = []

    for x_n in target_data:
        xcomp = len(x_n)+1
        e_n = input_data_point - np.dot(A,x_n) # Compute noise vector

        if np.linalg.norm(e_n,p_2) <= epsilon:  # Check if noise is below noiselevel
            # Project onto the null space of F
            proj_nullspace = np.dot(proj_ns_F, np.hstack((x_n, e_n)))[0:xcomp] # Project (x_n, e_n) onto nullspace of F, only take dim of x_n

            # Compute norm of feasible set based on projection onto null space
            norm_projection = np.linalg.norm(proj_nullspace, ord = p_1)   

            #add to diam_y 
            norm_projection_y.append(norm_projection)

            #get ascending diams
            if norm_projection > max_diam_Fy:
                max_diam_Fy = 2*norm_projection
                
    # obtain number of samples in feasible set (num_feas will be used for statistics later on)
    num_feas = len(norm_projection_y)        
    # get mean over diams 
    if num_feas > 0:      
        diameter_mean_y = np.mean(np.power(norm_projection_y,p))

    return diameter_mean_y, num_feas, max_diam_Fy

def worstcase_kernelsize(feasible_sets_list, p_1 ,p):
    """
    Computes the worst-case kernel size for noisy inverse problem from a list of feasible sets.

    Args:
        - feasible_sets_list: list of feasible sets.
        - p_1: Order of the norm on the target dataset $\mathcal{M}_1$. Set to p=2 for the ell 2 norm computation.
        - p: Order of the average kernel size. Set to p=2 for the MSE lower bound computation and p=1 for MAE lower bound computation.
    Returns:
        - worstcase_kersize: worst-case kernel size for a set of input data samples.
    """
    
    worstcase_kersize = 0

    for feasible_set_y in feasible_sets_list:
        feasible_set_y 
        # compute diameter of feasible set for one input data point
        diameter_mean_y, num_feas, max_diam_Fy = diams_feasibleset(feasible_set_y, p_1 ,p)
        if max_diam_Fy > worstcase_kersize:
            worstcase_kersize = max_diam_Fy
    
    return worstcase_kersize

def worstcase_kernelsize_sym(A, input_data, target_data, p_1, p_2, p, epsilon):
    """
    Computes the worst-case kernel size for noisy inverse problem with a linear forward model with additive noise, target and input data.

    Args:
        - A: The matrix (for which we are computing the Moore-Penrose inverse) of the inverse problem input_data = A(target_data)+noise.
        - input_data: Input data for an approximate inverse method.
        - target_data: Target or ground truth data for an approximate inverse method.
        - p_1: Order of the norm on the target dataset $\mathcal{M}_1$. Set to p=2 for the ell 2 norm computation.
        - p_2: Order of the norm on the target dataset $\mathcal{M}_2 = A(\mathcal{M}_1)+\mathcal{E}$. Set to p=2 for the ell 2 norm computation.
        - p: Order of the average kernel size. Set to p=2 for the MSE lower bound computation and p=1 for MAE lower bound computation.
        - epsilon: Noise level in the inverse problem input_data = A(target_data)+noise.
    
    Returns:
        - worstcase_kersize: worst-case kernel size for a set of input data samples.
    """
    worstcase_kersize =0

    for y in input_data:
        # compute diameter of feasible set for one input data point
        diameter_mean_y, num_feas, max_diam_Fy = diams_feasibleset_linear_forwardmodel_sym(A, y, target_data, p_1, p_2, p, epsilon)
        if max_diam_Fy > worstcase_kersize:
            worstcase_kersize = max_diam_Fy
    
    return worstcase_kersize

def average_kernelsize(feasible_sets_list, p_1, p):
    """
    Computes the average kernel size for noisy inverse problem from a list of feasible sets.

    Args:
        - feasible_sets_list: list of feasible sets.
        - p_1: Order of the norm on the target dataset $\mathcal{M}_1$. Set to p=2 for the ell 2 norm computation.
        - p: Order of the average kernel size. Set to p=2 for the MSE lower bound computation and p=1 for MAE lower bound computation.
    
    Returns:
        - average_kersize: worst-case kernel size for a set of input data samples.
    """
   
    average_kersize = 0
    num_samples = len(feasible_sets_list)

    for feasible_set_y in feasible_sets_list:
        # compute diameter of feasible set for one input data point (num_feas will be used for statistics later on)
        diameter_mean_y, num_feas, max_diam_Fy = diams_feasibleset(feasible_set_y, p_1 ,p)
        #add diameters means for obtaining average kersize to the power p
        average_kersize = average_kersize + diameter_mean_y
        
    # get mean over input data
    if average_kersize>0 and num_samples > 0:
        average_kersize = np.divide(average_kersize, num_samples)   
    else: 
        average_kersize = 0
    # take power 1/p to obtain average kersize
    average_kersize =  np.power(average_kersize, 1/p)
    
    return average_kersize

def average_kernelsize_sym(A, input_data, target_data, p_1, p_2, p, epsilon):
    """
    Computes the average kernel size for noisy inverse problem with a linear forward model with additive noise, target and input data.

    Args:
        - A: The matrix (for which we are computing the Moore-Penrose inverse) of the inverse problem input_data = A(target_data)+noise.
        - input_data: Input data for an approximate inverse method.
        - target_data: Target or ground truth data for an approximate inverse method.
        - p_1: Order of the norm on the target dataset $\mathcal{M}_1$. Set to p=2 for the ell 2 norm computation.
        - p_2: Order of the norm on the target dataset $\mathcal{M}_2 = A(\mathcal{M}_1)+\mathcal{E}$. Set to p=2 for the ell 2 norm computation.
        - p: Order of the average kernel size. Set to p=2 for the MSE lower bound computation and p=1 for MAE lower bound computation.
        - epsilon: Noise level in the inverse problem input_data = A(target_data)+noise.
    
    Returns:
        - average_kersize: average kernel size for a set of input data samples.
    """

    average_kersize_sym = 0
    num_samples = len(input_data)

    for y in input_data:
        # compute diameter of feasible set for one input data point (num_feas will be used for statistics later on)
        diameter_mean_y, num_feas, max_diam_Fy =  diams_feasibleset_linear_forwardmodel_sym(A, y, target_data, p_1, p_2, p, epsilon)
        #add diameters means for obtaining average kersize to the power p
        average_kersize_sym = average_kersize_sym + diameter_mean_y
        
    # get mean over input data
    if average_kersize_sym > 0 and num_samples >0: 
        average_kersize_sym = np.divide(average_kersize_sym, num_samples)    
    else:
        average_kersize_sym = 0
    # take power 1/p to obtain average kersize
    average_kersize_sym =  np.power(average_kersize_sym, 1/p)
    
    return average_kersize_sym

