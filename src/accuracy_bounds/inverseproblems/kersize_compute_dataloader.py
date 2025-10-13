from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm



def worstcase_kernelsize_appartenance(distsXX, feasible_appartenance):
    """
    Computes the worst-case kernelsize by computing the maximum pairwise distance of elements in feasible sets for all target samples.
    The feasible sets are handed over pairwise indices in a feasible appartenance matrix.

    Args:
        distsXX: (scipy.sparse matrix) containing pairwise distances between target samples.
        feasible_appartenance: (torch.ndarray or sparse matrix) matric of allocation of pairwise data indices to feasible sets.

    Returns:
        float: worst-case kernelsize.
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
    #distsXX = torch.asarray(distsXX.to_dense())

    results = Parallel(n_jobs=-1)(delayed(compute_max_distance)(y_idx, feasible_appartenance, distsXX) for y_idx in tqdm(range(p)))
    

    return np.nanmax(np.array(list(results)))

def average_kernelsize_appartenance(distsXX, feasible_appartenance,p_X):
    """
    Computes the average kernelsize from pairwise distances of elements in feasible sets for all target samples.
    The feasible sets are handed over pairwise indices in a feasible appartenance matrix.

    Args:
        distsXX: (scipy.sparse matrix) containing pairwise distances between target samples.
        feasible_appartenance: (torch.ndarray or sparse matrix) matric of allocation of pairwise data indices to feasible sets.
        p_X (int or float): Norm degree for distance computation.

    Returns:
        float: average kernelsize.
    """
    def compute_mean_distance(y_idx, feasible_appertinance_matrix, dXX):
        # Extract valid indices where feasible_appartenance[:, y_idx] is non-zero
        valid_idx = feasible_appertinance_matrix[:, y_idx].nonzero()[0]
        
        if len(valid_idx) == 0:
            return 0
        
        # Extract the submatrix from distsXX using valid indices
        subdistXX = dXX[valid_idx, :][:, valid_idx]
        subdistXX = subdistXX.toarray()  # Convert to dense matrix for max computation
        
        size_feas = len(valid_idx)
        # compute average over sums of differences over twice the feasible sets 
        return np.divide(np.nansum(np.power(subdistXX,p_X)), np.power(size_feas,2))


    n,p = feasible_appartenance.shape
    results = Parallel(n_jobs=-1)(delayed(compute_mean_distance)(y_idx, feasible_appartenance, distsXX) for y_idx in tqdm(range(p)))

    # get average over K input data samples (y) 
    return np.power(np.nanmean(np.asarray(list(results))),np.divide(1,p_X))



    
 
