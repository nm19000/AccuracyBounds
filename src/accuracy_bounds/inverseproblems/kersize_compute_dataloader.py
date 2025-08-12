import numpy as np
#from utils import projection_nullspace_operator
import torch
from scipy.sparse import csr_matrix, lil_matrix
from joblib import Parallel, delayed
from pdb import set_trace


def compute_feasible_set(A, input_data_point, target_data, p, epsilon):
    """
    Implements the iterative algorithm for estimating feasible sets for one input point based on possible target data points for a noisy inverse problem. 
        Arguments:
        - A: The matrix (for which we are computing the Moore-Penrose inverse) of the inverse problem input_data = A(target_data)+noise.
        - input_data_point: Input data point, referred to as "y" in variable names, for an approximate inverse method.
        - target_data: Target or ground truth data for an approximate inverse method.
        - p: order of the norm, default p=2 for MSE computation.
        - epsilon: Noise level in the inverse problem input_data = A(target_data)+noise.

        Returns:
        - feas_set_approx: approximation to the feasible set, consisting of all target data points that can map to an input data point within the noise level.
    """

    # Step 2: Compute feasible set for input data point y
    feas_set_y = []

    for x_n in target_data:
        e_n = input_data_point - np.dot(A,x_n) # Compute noise vector

        if np.linalg.norm(e_n,p) <= epsilon:  # Check if noise is below noiselevel
            # add traget data point x_n to feasible set
            feas_set_y.append(x_n)

    return feas_set_y

def diams_feasibleset_inv(A, input_data_point, target_data, p, epsilon):
    """
    Implements the iterative algorithm for diameter estimation of the feasible set for a noisy inverse problem. 
    Computes diameter based on possible target data points for one input point.
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

    # Step 2: Compute feasible sets and diameters
    feas_set_y = compute_feasible_set(A, input_data_point, target_data, p, epsilon)
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
            diameter= np.linalg.norm(dist_ns, ord = p)

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

def diams_feasibleset_inv_sym(A, input_data_point, target_data, p, epsilon):
    """
    Implements the iterative algorithm for diameter estimation of the feasible set for a noisy inverse problem under the assumption that 
    there exists noise vectors such that the target data is symmetric with respect to the null space of F= (A |I): P_{N(F)^perp}(x,e)-P_{N(F)}(x,e). 
    Computes diameter based on possible target data points for one input point.
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
    proj_ns_F = projection_nullspace_operator(F)

    # Step 2: Compute feasible sets and diameters
    max_diam_Fy = 0
    diameter_mean_y = 0
    diam_y = []

    for x_n in target_data:
        xcomp = len(x_n)-1
        e_n = input_data_point - np.dot(A,x_n) # Compute noise vector

        if np.linalg.norm(e_n,p) <= epsilon:  # Check if noise is below noiselevel
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

def wc_kernelsize(A, input_data, target_data, p, epsilon):
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

def av_kernelsize(A, input_data, target_data, p, epsilon):
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

def wc_kernelsize_sym_crossbatch_cuda(A,F_null, batch1, batch2, p_X, p_Y, epsilon):

    input1, target1 = batch1
    input2,target2 = batch2

    return wc_kernelsize_sym_batch_cuda(A, F_null, input1, target2, p_X = p_X, p_Y = p_Y, epsilon = epsilon )

def wc_kernelsize_nosym_crossbatch_cuda(A, batch1, batch2, p_X, p_Y, epsilon):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input1, target1 = batch1
    input2,target2 = batch2

    y1 = torch.tensor(input1, dtype=torch.float32, device=device)
    n1 = y1.shape[0]
    y1_flat = y1.reshape(n1, -1)

    y2 = torch.tensor(input2, dtype=torch.float32, device=device)
    n2 = y2.shape[0]
    y2_flat = y2.reshape(n1, -1)

    cross_dist = torch.norm(y1_flat[:,None,:]-y2_flat[None,:,:], dim = -1, p = p_Y)
    same_feasible = cross_dist< 2*epsilon

    x1 = torch.tensor(target1, dtype=torch.float32, device=device)
    x1_flat = x1.reshape(n1, -1)

    x2 = torch.tensor(target2, dtype=torch.float32, device=device)
    x2_flat = x2.reshape(n1, -1)

    target_dists = torch.norm(x1_flat[:,None,:]-x2_flat[None,:,:], dim = -1, p = p_X)
    masked_target_dist = torch.where(same_feasible, target_dists, torch.tensor(float('nan'), device=device))

    return np.nanmax(masked_target_dist.cpu().numpy())

def wc_kernelsize_sym_perbatch_cuda(A, F_null, input_data, target_data, p_X, p_Y, epsilon, batch_size):
    '''
    forwarded input has to be target_data@(A.T) , or equivalent. 
    It is frequently the same as input_data, but in the description of the algorithm, that is not necessary
    '''
    p = input_data.shape[0]
    n_batches = p//batch_size

    if F_null is None:
        # Compute Moore-Penrose-Inverse of F
        F = np.hstack((A, np.eye(A.shape[0])))  # Construct F: (A | I) 
        F_null = projection_nullspace_operator(F)  # to replace by the one in my class


    current_kersize = 0
    for i in range(n_batches):
        idx_imin = i*batch_size
        idx_imax = min(idx_imin+ batch_size, p)
        
        batch_i_current = (input_data[idx_imin:idx_imax], target_data[idx_imin:idx_imax])
        for j in range(n_batches):
            idx_jmin = j*batch_size
            idx_jmax = min(idx_jmin+ batch_size, p)

            batch_j_current = (input_data[idx_jmin:idx_jmax], target_data[idx_jmin:idx_jmax])
            
      
            if i==j:
                ks_batch = wc_kernelsize_sym_batch_cuda(A,F_null, batch_i_current[0], batch_i_current[1], p_X = p_X, p_Y = p_Y, epsilon=epsilon)
            else:
                ks_batch = wc_kernelsize_sym_crossbatch_cuda(A, batch_i_current, batch_j_current, p_X = p_X, p_Y = p_Y, epsilon=epsilon)
            if ks_batch >current_kersize:
                current_kersize = ks_batch
    return current_kersize

def wc_kernelsize_nosym_perbatch_cuda(A, input_data, target_data, p_X, p_Y, epsilon, batch_size):
    p = input_data.shape[0]
    n_batches = p//batch_size

    current_kersize = 0
    for i in range(n_batches):
        idx_imin = i*batch_size
        idx_imax = min(idx_imin+ batch_size, p)
        
        batch_i_current = (input_data[idx_imin:idx_imax], target_data[idx_imin:idx_imax])
        for j in range(i,n_batches):
            idx_jmin = j*batch_size
            idx_jmax = min(idx_jmin+ batch_size, p)

            batch_j_current = (input_data[idx_jmin:idx_jmax], target_data[idx_jmin:idx_jmax])
            
      
            if i==j:
                ks_batch = wc_kernelsize_nosym_batch_cuda(A, batch_i_current[0], batch_i_current[1], p_X = p_X, p_Y = p_Y, epsilon=epsilon)
            else:
                ks_batch = wc_kernelsize_nosym_crossbatch_cuda(A, batch_i_current, batch_j_current, p_X = p_X, p_Y = p_Y, epsilon=epsilon)
            if ks_batch >current_kersize:
                current_kersize = ks_batch
    return current_kersize

def wc_kernelsize_sym_batch_cuda(A, F_null, input_data, target_data, p_X, p_Y, epsilon):
    '''
    forwarded input has to be target_data@(A.T) , or equivalent. 
    It is frequently the same as input_data, but in the description of the algorithm, that is not necessary
    '''
    if F_null is None:
        # Compute Moore-Penrose-Inverse of F
        F = np.hstack((A, np.eye(A.shape[0])))  # Construct F: (A | I) 
        F_null = projection_nullspace_operator(F)  # to replace by the one in my class

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(target_data, dtype=torch.float32, device=device)
    p = x.shape[0]
    n = x.shape[1]
    x_flat = x.reshape(p, -1)

    y = torch.tensor(input_data, dtype=torch.float32, device=device)
    q = y.shape[0]
    y_flat = y.reshape(q, -1)

    A_t = torch.tensor(A, dtype = torch.float32, device = device).T
    e_diff = (-x_flat@A_t)[:,None, :] + y_flat[None, :,:] # Calculating e_{i,j} =  y_i- Ax_j for every i,j
    feasible_appartenance = torch.norm(e_diff,p = p_Y, dim = -1)<epsilon # x_j belongs in F_{y_i} iff feasible_appartenance[j,i] is true
    feasible_appartenance = feasible_appartenance.unsqueeze(-1)
    
    nanvals = torch.full_like(e_diff,float('nan'), device=device)
    e_diff = torch.where(feasible_appartenance, e_diff, nanvals) # Masking with nan (so that it can be reused easily for the average kernel size)
    
    x_exp = x_flat[:,None, :].expand(-1, q, -1)
    x_e_concat = torch.cat([x_exp, e_diff], dim = -1)  # Concatenating every x_i to each e_{i,j} 
    nm = x_e_concat.shape[-1]
 
    Fy_projs = x_e_concat.reshape(-1, nm)@(F_null.T) # Projectng each concatenation if they are in the feasible set. (if they are not, the concatenation will contain some nan and the operation will not be possible)
    Fy_projs = Fy_projs.reshape(p,q,nm)[:,:,:n]  # Only take the x part

    # it seems that torch.einsum('pqd,md->pqm', x_e_concat, F_null) coule be used, but i don't master it. I mention it because it looks fancy
    target_dists = 2*torch.norm(Fy_projs, p = p_X,dim = -1)
    return np.nanmax(target_dists.cpu().numpy()) 

def wc_kernelsize_nosym_batch_cuda(A, input_data, target_data, p_X, p_Y, epsilon):
    # Convert input_data to torch tensor and move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(input_data, dtype=torch.float32, device=device)
    n = x.shape[0]
    # If x has shape (n, w, h, c, p), compute pairwise distances over all dimensions except the first
    # Flatten all but the first dimension for distance computation
    x_flat = x.reshape(n, -1)  # shape (n, D)
    diff = x_flat.unsqueeze(1) - x_flat.unsqueeze(0)  # shape (n, n, D)
    dist = torch.norm(diff, p=p_Y, dim=-1)  # shape (n, n)

    same_feasible = dist<2*epsilon

    # Now compute pairwise distances for target_data, but only where same_feasible is True
    target = torch.tensor(target_data, dtype=torch.float32, device=device)
    target_flat = target.reshape(target.shape[0], -1)  # shape (n, D)
    target_diff = target_flat.unsqueeze(1) - target_flat.unsqueeze(0)  # shape (n, n, D)
    target_dist = torch.norm(target_diff, p=p_X, dim=-1)  # shape (n, n)

    # Mask target_dist with same_feasible, use nan instead of 0
    masked_target_dist = torch.where(same_feasible, target_dist, torch.tensor(float('nan'), device=device))
    return np.nanmax(masked_target_dist.cpu().numpy())

'''
We distinguish 2 kind of samplings : 
- The first is that we sample int the Y space the input_data and we look at which x is in which feasible set. 
    This one is denoted by samplingYX

- The second one is that we sample in the X space only, and the forward it without noise (if possible, or at least without explicit noise vector) into the y space.
    Then, we check which x,x' belong to a common feasible set.
    This one is denoted by samplingX


'''


def target_distances_samplingYX_perbatch_cuda(A, input_data, target_data, forwarded_target, p_X, p_Y, epsilon, batch_size):
    """
    Computes pairwise distances between target samples (in X), considering only those pairs that belong to the same feasible set F_y.
    Splits data into batches for efficient computation and constructs a matrix indicating which x belongs to which F_y.

    Args:
        input_data (np.ndarray): Input data samples (y).
        target_data (np.ndarray): Target data samples (x).
        forwarded_target (np.ndarray): Forwarded target data (F(x,e)).
        p_X (int or float): Norm degree for target distance (X) computation.
        p_Y (int or float): Norm degree for feasibility computation.
        epsilon (float): Noise level in the model.
        batch_size (int): Size of each batch for computation.

    Returns:
        distsXX (scipy.sparse.lil_matrix): Matrix of pairwise distances between feasible target samples.
        feasible_appartenance (scipy.sparse.csr_matrix): Feasibility matrix indicating which x belongs to which F_y.
    """


    n_batches_input = len(input_data.dataset)
    n_batches_target = len(target_data.dataset)

    feasible_appartenance = lil_matrix((m, n), dtype=np.float32)

    for target_batch_id, target_batch in enumerate(target_data):

        for input_batch_id, input_batch in enumerate(input_data):
            feasible_small = feasibleApp_samplingYX_batch_cuda(A, input_batch["img"], target_batch["img"], p_Y, epsilon)
            feasible_small = csr_matrix(feasible_small)

            batch_size_target = target_batch["img"].shape[0]
            batch_size_input = input_batch["img"].shape[0]

            # TODO: what if batchsize is not always the same? e.g. drop_last = False
            idx_imin = batch_size_target * target_batch_id
            idx_imax = batch_size_target * (target_batch_id + 1)

            idx_jmin = batch_size_input * input_batch_id
            idx_jmax = batch_size_input * (input_batch_id + 1)


            feasible_appartenance[idx_imin:idx_imax, idx_jmin:idx_jmax] = feasible_small

    feasible_appartenance = feasible_appartenance.tocsr()

    
    #print(feasible_appartenance.data.nbytes/(1024*1024), f'size data {feasible_appartenance.shape}') 
    
    common_feasible = feasible_appartenance@(feasible_appartenance.T)

    distsXX = lil_matrix((m, m), dtype=np.float32)
    
    # TODO: Why iterate this a second time? Why not computed in first loop as well? 
    for i in range(n_batches_target):
        idx_imin = i*batch_size
        idx_imax = idx_imin+ batch_size

        for j in range(n_batches_target):
            idx_jmin = j*batch_size
            idx_jmax = idx_jmin + batch_size


            dists_small = np.array(distsXX_samplingYX_batch_cuda(A, target_data[idx_imin:idx_imax], target_data[idx_jmin:idx_jmax], p_X))
            
            common_feasible_small = common_feasible[idx_imin:idx_imax, idx_jmin:idx_jmax].toarray()
            mask = common_feasible_small != 0

            dists_small[~mask] = 0
            dists_small = csr_matrix(dists_small)

            distsXX[idx_imin:idx_imax, idx_jmin:idx_jmax] = dists_small

    return distsXX, feasible_appartenance

def distsXX_samplingYX_batch_cuda(A, target_data1, target_data2 , p_X):
    """
    Computes pairwise distances between two batches of target data (x) using the specified norm.

    Args:
        target_data1 (np.ndarray): First batch of target data.
        target_data2 (np.ndarray): Second batch of target data.
        p_X (int or float): Norm degree for distance computation.

    Returns:
        torch.Tensor: Matrix of pairwise distances between target_data1 and target_data2.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x1 = torch.tensor(target_data1, dtype = torch.float32,device = device )
    x2 = torch.tensor(target_data2, dtype = torch.float32,device = device )

    n1 = x1.shape[0]
    n2 = x2.shape[0]

    x1_flat = torch.tensor(x1.reshape(n1, -1), device = device)
    x2_flat = torch.tensor(x2.reshape(n2, -1), device = device)

    distancesXX = torch.norm(x1_flat[:,None, :]-x2_flat[None,:,:], p = p_X, dim = -1) 
    return distancesXX

def feasibleApp_samplingYX_batch_cuda(A, input_data,forwarded_target, p_Y, epsilon):
    """
    Determines which target samples belong to the feasible set of each input data sample.

    Args:
        input_data (np.ndarray): Batch of input samples (y).
        forwarded_target (np.ndarray): Batch of forwarded target samples (F(x,e)).
        p_Y (int or float): Norm degree for feasibility computation.
        epsilon (float): Feasibility threshold.

    Returns:
        np.ndarray: Boolean matrix indicating feasibility (shape: [forwarded_target, input_data]).
    """
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forwarded_target = torch.tensor(forwarded_target, dtype = torch.float32, device = device)
    input_data = torch.tensor(input_data, dtype = torch.float32, device = device)

    n = input_data.shape[0]
    m = forwarded_target.shape[0]
    input_data_flat = torch.tensor(input_data.reshape(n,-1), device = device)
    forwarded_flat = torch.tensor(forwarded_target.reshape(m,-1),device = device)

    e_diff = -forwarded_flat[:,None, :] + input_data_flat[None,:,:]

    feasible_appartenance = np.asarray(torch.norm(e_diff,p = p_Y, dim = -1)<epsilon)

    return feasible_appartenance # feasible appartenance is y vs x

def kersize_samplingYX(distsXX, feasible_appartenance,p_X):
    """
    Computes the Kersize approximation as the maximum pairwise distance within feasible sets for all target samples.

    Args:
        distsXX (scipy.sparse matrix): Pairwise distance matrix between target samples.
        feasible_appartenance (np.ndarray or sparse matrix): Feasibility matrix.
        p_X (int or float): Norm degree for distance computation.

    Returns:
        float: Kersize.
    """
    def compute_max_distance(y_idx, fa, dXX):
        # Extract valid indices where feasible_appartenance[:, y_idx] is non-zero
        valid_idx = fa[:, y_idx].nonzero()[0]
        
        if len(valid_idx) == 0:
            return 0
        
        # Extract the submatrix from distsXX using valid indices
        subdistXX = dXX[valid_idx, :][:, valid_idx]
        subdistXX = subdistXX.toarray()  # Convert to dense matrix for max computation
        
        # Compute the maximum distance
        distmax = np.nanmax(subdistXX)
        return distmax
    

    n,p = feasible_appartenance.shape
    #distsXX = np.asarray(distsXX.to_dense())

    results = Parallel(n_jobs=-1)(delayed(compute_max_distance)(y_idx, feasible_appartenance, distsXX) for y_idx in range(p))
    

    return np.nanmax(np.array(list(results)))

def avgLB_samplingYX(distsXX, feasible_appartenance, p_X):
    """
    Computes the lower bound of the average error.

    Args:
        distsXX (scipy.sparse matrix): Pairwise distance matrix between target samples (x).
        feasible_appartenance (np.ndarray or sparse matrix): Feasibility matrix.
        p_X (int or float): Norm degree for distance computation.

    Returns:
        float: Average lower bound of feasible pairwise distances.
    """
    def compute_mean_distance(y_idx, fa, dXX):
        # Extract valid indices where feasible_appartenance[:, y_idx] is non-zero
        valid_idx = fa[:, y_idx].nonzero()[0]
        
        if len(valid_idx) == 0:
            return 0
        
        # Extract the submatrix from distsXX using valid indices
        subdistXX = dXX[valid_idx, :][:, valid_idx]
        subdistXX = subdistXX.toarray()  # Convert to dense matrix for max computation
        
        size_feas = len(valid_idx)
        return 2*np.nanmean(subdistXX**p_X)



    n,p = feasible_appartenance.shape
    results = Parallel(n_jobs=-1)(delayed(compute_mean_distance)(y_idx, feasible_appartenance, distsXX) for y_idx in range(p))

    return np.nanmean(np.asarray(list(results)))/(2**p_X)

def avgkersize_samplingYX(distsXX, feasible_appartenance, p_X):
    """
    Computes the average kernel size approcimation.

    Args:
        distsXX (scipy.sparse matrix): Pairwise distance matrix between target samples.
        feasible_appartenance (np.ndarray or sparse matrix): Feasibility matrix.
        p_X (int or float): Norm degree for distance computation.

    Returns:
        float: Average kernel size.
    """
    def compute_mean_distance(y_idx, fa, dXX):
        # Extract valid indices where feasible_appartenance[:, y_idx] is non-zero
        valid_idx = fa[:, y_idx].nonzero()[0]
        
        if len(valid_idx) == 0:
            return 0
        
        # Extract the submatrix from distsXX using valid indices
        subdistXX = dXX[valid_idx, :][:, valid_idx]
        subdistXX = subdistXX.toarray()  # Convert to dense matrix for max computation
        
        size_feas = len(valid_idx)
        return 2*np.nanmean(subdistXX**p_X)
    


    n,p = feasible_appartenance.shape
    results = Parallel(n_jobs=-1)(delayed(compute_mean_distance)(y_idx, feasible_appartenance, distsXX) for y_idx in range(p))

 
    return np.nanmean(np.asarray(results))**(1/p_X)



def target_distances_samplingX_batch_cuda(A,input_data, target_data, p_X, p_Y, epsilon):
    """
    Computes pairwise distances between target samples, masking out pairs whose corresponding inputs are not in some common feasible set.

    Args:
        input_data (np.ndarray): Input data samples.
        target_data (np.ndarray): Target data samples.
        p_X (int or float): Norm degree for target distance computation.
        p_Y (int or float): Norm degree for feasibility computation.
        epsilon (float): Feasibility threshold.

    Returns:
        np.ndarray: Masked pairwise distance matrix for target samples.
        torch.Tensor: Boolean matrix indicating feasibility between input samples.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(input_data, dtype=torch.float32, device=device)
    n = x.shape[0]
    # If x has shape (n, w, h, c, p), compute pairwise distances over all dimensions except the first
    # Flatten all but the first dimension for distance computation
    x_flat = x.reshape(n, -1)  # shape (n, D)
    diff = x_flat.unsqueeze(1) - x_flat.unsqueeze(0)  # shape (n, n, D)
    dist = torch.norm(diff, p=p_Y, dim=-1)  # shape (n, n)

    same_feasible = dist<2*epsilon

    # Now compute pairwise distances for target_data, but only where same_feasible is True
    target = torch.tensor(target_data, dtype=torch.float32, device=device)
    target_flat = target.reshape(target.shape[0], -1)  # shape (n, D)
    target_diff = target_flat.unsqueeze(1) - target_flat.unsqueeze(0)  # shape (n, n, D)
    target_dist = torch.norm(target_diff, p=p_X, dim=-1)  # shape (n, n)

    masked_target_dist = torch.where(same_feasible, target_dist, torch.tensor(float('nan'), device=device))

    # Convert masked_target_dist to numpy array
    masked_target_dist_np = masked_target_dist.cpu().numpy()
 

    return masked_target_dist_np, same_feasible

def target_distances_samplingX_crossbatch_cuda(A, batch1, batch2, p_X, p_Y, epsilon):
    """
    Computes pairwise distances between target samples from two batches, masking out pairs whose corresponding inputs are not feasible.

    Args:
        batch1 (tuple): Tuple of (input_data, target_data) for batch 1.
        batch2 (tuple): Tuple of (input_data, target_data) for batch 2.
        p_X (int or float): Norm degree for target distance computation.
        p_Y (int or float): Norm degree for feasibility computation.
        epsilon (float): Feasibility threshold.

    Returns:
        np.ndarray: Masked pairwise distance matrix for target samples.
        torch.Tensor: Boolean matrix indicating feasibility between input samples.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input1, target1 = batch1
    input2,target2 = batch2

    y1 = torch.tensor(input1, dtype=torch.float32, device=device)
    n1 = y1.shape[0]
    y1_flat = y1.reshape(n1, -1)

    y2 = torch.tensor(input2, dtype=torch.float32, device=device)
    n2 = y2.shape[0]
    y2_flat = y2.reshape(n1, -1)

    cross_dist = torch.norm(y1_flat[:,None,:]-y2_flat[None,:,:], dim = -1, p = p_Y)
    same_feasible = cross_dist< 2*epsilon

    x1 = torch.tensor(target1, dtype=torch.float32, device=device)
    x1_flat = x1.reshape(n1, -1)

    x2 = torch.tensor(target2, dtype=torch.float32, device=device)
    x2_flat = x2.reshape(n1, -1)

    target_dists = torch.norm(x1_flat[:,None,:]-x2_flat[None,:,:], dim = -1, p = p_X)
    masked_target_dist = torch.where(same_feasible, target_dists, torch.tensor(float('nan'), device=device))
    

    # Convert masked_target_dist to numpy array
    masked_target_dist_np = masked_target_dist.cpu().numpy()

    return masked_target_dist_np, same_feasible

def target_distances_samplingX_perbatch_cuda(A, input_data, target_data, p_X, p_Y, epsilon, batch_size):
    """
    Computes pairwise distances between target samples in batches, masking out pairs whose corresponding inputs are not in common feasible sets.

    Args:
        input_data (np.ndarray): Input data samples.
        target_data (np.ndarray): Target data samples.
        p_X (int or float): Norm degree for target distance computation.
        p_Y (int or float): Norm degree for feasibility computation.
        epsilon (float): Feasibility threshold.
        batch_size (int): Size of each batch for computation.

    Returns:
        np.ndarray: Masked pairwise distance matrix for target samples.
        np.ndarray: Boolean matrix indicating feasibility between input samples.
    """
    n = target_data.shape[0]
    n_batches = (n//batch_size)

    masked_target_dists = np.zeros((n,n))
    same_feasible = np.zeros((n,n))

    for i in range(n_batches):
        idx_imin = i*batch_size
        idx_imax = min(idx_imin+ batch_size, n)
        
        batch_i_current = (input_data[idx_imin:idx_imax], target_data[idx_imin:idx_imax])
        for j in range(i,n_batches):
            idx_jmin = j*batch_size
            idx_jmax = min(idx_jmin+ batch_size, n)

            batch_j_current = (input_data[idx_jmin:idx_jmax], target_data[idx_jmin:idx_jmax])
            
      
            if i==j:
                masked_target_dists_local, same_feasible_local = target_distances_samplingX_batch_cuda(A,batch_i_current[0], batch_i_current[1], p_X, p_Y, epsilon )
                masked_target_dists[idx_imin:idx_imax, idx_jmin:idx_jmax] = masked_target_dists_local
                same_feasible[idx_imin:idx_imax, idx_jmin:idx_jmax] = same_feasible_local

            else:
                masked_target_dists_local, same_feasible_local = target_distances_samplingX_crossbatch_cuda(A, batch_i_current, batch_j_current, p_X, p_Y, epsilon)
                masked_target_dists[idx_imin:idx_imax, idx_jmin:idx_jmax] = masked_target_dists_local
                same_feasible[idx_imin:idx_imax, idx_jmin:idx_jmax] = same_feasible_local

    return masked_target_dists, same_feasible

def kersize_samplingX(masked_target_dists):
    """
    Computes the kernelsize.

    Args:
        masked_target_dists (np.ndarray): Masked pairwise distance matrix for target samples.

    Returns:
        float: Kersize.
    """
    return np.nanmax(masked_target_dists)

