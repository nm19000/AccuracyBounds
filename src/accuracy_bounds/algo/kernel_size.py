import numpy as np
from .feasibility import diams_feasibleset_inv, diams_feasibleset_inv_sym

def compute_av_kernel_size(A, input_data, target_data, p, q, epsilon, max_k):

    av_kersize_list = av_kernelsize(A, input_data, target_data, p,q, epsilon, max_k)
    
    print("AV-Kernel Size:", av_kersize_list[-1])

    return np.array(av_kersize_list)

def compute_wc_kernel_size(A, input_data, target_data, p, q, epsilon, max_k):

    wc_kersize_k_list = wc_kernelsize(A, input_data, target_data, p,q, epsilon, max_k)
    
    print("WC-Kernel Size:", wc_kersize_k_list[-1])

    return np.array(wc_kersize_k_list)

def wc_kernelsize(A, input_data, target_data, p,q, epsilon, max_k):
    """
    Computes the worst-case kernel size for noisy inverse problem using Algorithm 1.

    Args:
        - A: The matrix (for which we are computing the Moore-Penrose inverse) of the inverse problem input_data = A(target_data)+noise.
        - input_data: Input data for an approximate inverse method.
        - target_data: Target or ground truth data for an approximate inverse method.
        - p: order of the norm on the target dataset, default p=2 for MSE computation.
        - q: order of norm on the input dataset.
        - epsilon: Noise level in the inverse problem input_data = A(target_data)+noise.

    Returns:
        Approximate worst-case kernel size for a set of input data samples.
    """

    wc_kersize_list = []
    k = 1

    if isinstance(input_data, np.ndarray) and input_data.ndim==2:
        # Adjust to batchsize 1
        input_data = input_data.reshape([input_data.shape[0], 1, input_data.shape[1]])

    # compute diameter of feasible set for one input data point
    for input_data_batch in input_data: 
        for input_data_point in input_data_batch: 
            k += 1
            diameter, num_feas, wc_kersize = diams_feasibleset_inv(A, input_data_point, target_data, p, epsilon)
            wc_kersize_list.append(wc_kersize)
            
        if k >= max_k:
            break
    
    wc_kersize_list = wc_kersize_list[:max_k-1] # TODO I took this from the previous implementation. SHouldnt it be :max_k ? 

    for i in range(len(wc_kersize_list)-1):
        if wc_kersize_list[i+1] < wc_kersize_list[i]: 
            wc_kersize_list[i+1] = wc_kersize_list[i]

    return wc_kersize_list

def av_kernelsize(A, input_data, target_data, p,q, epsilon, max_k):
    """
    Computes the average kernel size for a noisy inverse problem under Algorithm 2.

    Args:
        - A: The matrix (for which we are computing the Moore-Penrose inverse) of the inverse problem input_data = A(target_data)+noise.
        - input_data: Input data for an approximate inverse method.
        - target_data: Target or ground truth data for an approximate inverse method.
        - p: order of the norm on the target dataset, default p=2 for MSE computation.
        - q: order of norm on the input dataset.
        - epsilon: Noise level in the inverse problem input_data = A(target_data)+noise.

    Returns:
        Approximate average kernel size for for a set of input data samples.
    """

    av_kersize_list = []
    k = 0

    if isinstance(input_data, np.ndarray) and input_data.ndim==2:
        # Adjust to batchsize 1
        input_data = input_data.reshape([input_data.shape[0], 1, input_data.shape[1]])

    for input_data_batch in input_data: 
        for input_data_point in input_data_batch: 
            k += 1        # compute diameter of feasible set for one input data point (num_feas will be used for statistics later on)
            diameter, num_feas, wc_kersize = diams_feasibleset_inv_sym(A, input_data_point, target_data, p,q, epsilon)
            #add diameters means for obtaining average kersize to the power p
            av_kersize_list.append(diameter)

        if k >= max_k: 
            break
    
    av_kersize_list = av_kersize_list[:max_k]
    av_kersize_list = np.cumsum(av_kersize_list) / np.arange(1, len(av_kersize_list)+1)

    # take power 1/p to obtain average kersize
    av_kersize_list =  np.power(av_kersize_list, 1/p)

    return av_kersize_list

def av_kersizes_over_samples_sizes(A, input_data_loads, target_data, p,q, epsilon):
    """Computes the average kernel size for a noisy inverse problem under Algorithm 2.
        Args:
            - A: The matrix (for which we are computing the Moore-Penrose inverse) of the inverse problem input_data = A(target_data)+noise.
            - input_data: Input data for an approximate inverse method.
            - target_data: Target or ground truth data for an approximate inverse method.
            - p: order of the norm on the target dataset, default p=2 for MSE computation.
            - q: order of norm on the input dataset.
            - epsilon: Noise level in the inverse problem input_data = A(target_data)+noise.
        Returns:
            Average kernel sizes for sample sizes of input data. """
    
    av_kersizes =[]

    for input_data in input_data_loads:
        av_kersize = av_kernelsize(A, input_data, target_data, p,q, epsilon)
        av_kersizes.append(av_kersize)
    
    return av_kersizes   


def wc_kersizes_over_samples_sizes(A, input_data_loads, target_data, p,q, epsilon):
    """Computes the worst-case kernel size for a noisy inverse problem under Algorithm 2.
        Args:
            - A: The matrix (for which we are computing the Moore-Penrose inverse) of the inverse problem input_data = A(target_data)+noise.
            - input_data: Input data for an approximate inverse method.
            - target_data: Target or ground truth data for an approximate inverse method.
            - p: order of the norm on the target dataset, default p=2 for MSE computation.
            - q: order of norm on the input dataset.
            - epsilon: Noise level in the inverse problem input_data = A(target_data)+noise.
        Returns:
            Worst-case kernel sizes for sample sizes of input data. """
    
    wc_kersizes =[]

    for input_data in input_data_loads:
        wc_kersize = wc_kernelsize(A, input_data, target_data, p,q, epsilon)
        wc_kersizes.append(wc_kersize)
    
    return wc_kersizes  