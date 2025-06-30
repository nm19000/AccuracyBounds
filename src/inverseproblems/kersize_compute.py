import numpy as np
from utils import projection_nullspace, apply_forwardmodel


def diameters_feasiblesets(A, input_data, target_data, p=2, epsilon=1e-1):
    """
    Implements the iterative algorithm for diameter estimation of the feasible set, consisting of all possible target data points, for each target point.
        Arguments:
        - A: The matrix (for which we are computing the Moore-Penrose inverse) of the inverse problem input_data = A(target_data)+noise.
        - input_data: Input data for an approximate inverse method.
        - target_data: Target or ground truth data for an approximate inverse method.
        - p: order of the norm, default p=2 for MSE computation.
        - epsilon: Noise level in the inverse problem input_data = A(target_data)+noise.

        Returns:
        - diameters: dim(0)= shape(input_data), the estimated diameters of the feasible set, consisting of all possible target data points, for each target point.
    """
    # Compute Moore-Penrose-Inverse of F
    F = np.hstack((A, np.eye(A.shape[0])))  # Construct F: (A | I) 

    # Step 2: Compute diameters
    diameter_means = []
    max_diameters = []

    for y in input_data:
        max_diam_F_y = 0
        diameter_mean_y = 0
        diam_y = []

        for x_n in target_data:
            xcomp = len(x_n)
            #y_extended = np.hstack((y, 0))  # Extend y_i to 3D
            e_n = y - np.dot(A,x_n) # Compute noise vector

            if np.linalg.norm(e_n,p) <= epsilon:  # Check if noise is in E
                # Project onto the null space of F
                proj_nullspace = projection_nullspace(F, np.hstack((x_n, e_n)))[0:xcomp] # Project (x_n, e_n) onto nullspace of F, only take dim of x_n
                #print(np.shape(proj_nullspace))

                # Compute diameter based on projection
                diameter = 2 * np.linalg.norm(proj_nullspace, ord = p)   

                #add to diam_y
                diam_y.append(diameter)

                #get ascending diams
                if diameter > max_diam_F_y:
                    max_diam_F_y = diameter
                
         # get mean over diams for one y       
        diameter_mean_y = np.mean(np.power(diam_y,p))
        # add mean diam_y to list of diams for av. kersize computation
        diameter_means.append(diameter_mean_y)
        # add max diam to list for wc kersize computation
        max_diameters.append(max_diam_F_y)
    max_diameter = np.max(max_diameters)

    return diameter_means, max_diameter

def wc_kernelsize_oversamples(A, input_data, target_data, p, max_k, epsilon=1e-1):
    """
    Computes the worst-case kernel size under noise using Algorithm 2.

    Args:
        - A: The matrix (for which we are computing the Moore-Penrose inverse) of the inverse problem input_data = A(target_data)+noise.
        - input_data: Input data for an approximate inverse method.
        - target_data: Target or ground truth data for an approximate inverse method.
        - p: order of the norm, default p=2 for MSE computation.
        - max_k (int): Maximum number of samples from traget data.
        - epsilon: Noise level in the inverse problem input_data = A(target_data)+noise.

    Returns:
        Approximate worst-case kernel size for k_max and array of approximate worst-case kernel sizes for k in range(max_k)
    """

    #max_diameters = []
    wc_kersize = []

    for k in range(1,max_k,1):
        input_data_k = input_data[0:k,:]
        diameter_means, max_diameter = diameters_feasiblesets(A, input_data_k, target_data, p, epsilon)
        #max_diameters.append(max_diameter)
        print(f"For k={k} we get wc_kersize={max_diameter}")
        wc_kersize.append(max_diameter)
        #k = k+1
    wc_kersizef=np.max(wc_kersize)  
    
    return wc_kersizef, wc_kersize

def av_kernelsize_oversamples(A, input_data, target_data, p, max_k, epsilon=1e-1):
    """
    Computes the average kernel size under noise using Algorithm 2.

    Args:
        - A: The matrix (for which we are computing the Moore-Penrose inverse) of the inverse problem input_data = A(target_data)+noise.
        - input_data: Input data for an approximate inverse method.
        - target_data: Target or ground truth data for an approximate inverse method.
        - p: order of the norm, default p=2 for MSE computation.
        - max_k (int): Maximum number of samples from traget data.
        - epsilon: Noise level in the inverse problem input_data = A(target_data)+noise.

    Returns:
        Approximate average kernel size for k_max and array of approximate average kernel sizes for k in range(max_k)
    """

    av_kersizes = []

    for k in range(1,max_k,1):
        input_data_k = input_data[0:k,:]
        diameter_means, max_diameter = diameters_feasiblesets(A, input_data_k, target_data, p, epsilon)
        #max_diameters.append(max_diameter)
        av_kersize =  np.power(np.mean(diameter_means), 1/p)
        av_kersizes.append(av_kersize)
        print(f"For k={k} we get av_kersize={av_kersize}")
        #k = k+1 
    av_kersize = av_kersizes[-1]
    
    return av_kersize, av_kersizes