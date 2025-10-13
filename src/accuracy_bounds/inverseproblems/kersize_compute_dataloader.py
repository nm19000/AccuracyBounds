from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm



def kersize_samplingYX(distsXX, feasible_appartenance):
    """
    Computes the Kersize approximation as the maximum pairwise distance within feasible sets for all target samples.

    Args:
        distsXX (scipy.sparse matrix): Pairwise distance matrix between target samples.
        feasible_appartenance (torch.ndarray or sparse matrix): Feasibility matrix.
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
    #distsXX = torch.asarray(distsXX.to_dense())

    results = Parallel(n_jobs=-1)(delayed(compute_max_distance)(y_idx, feasible_appartenance, distsXX) for y_idx in tqdm(range(p)))
    

    return np.nanmax(np.array(list(results)))

def avgLB_samplingYX(distsXX, feasible_appartenance,p_X):
    """
    Computes the lower bound of the average error.

    Args:
        distsXX (scipy.sparse matrix): Pairwise distance matrix between target samples (x).
        feasible_appartenance (torch.ndarray or sparse matrix): Feasibility matrix.
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
        return np.divide(np.nansum(np.power(subdistXX,p_X)), np.power(size_feas,2))


    n,p = feasible_appartenance.shape
    results = Parallel(n_jobs=-1)(delayed(compute_mean_distance)(y_idx, feasible_appartenance, distsXX) for y_idx in tqdm(range(p)))

    return np.power(np.nanmean(np.asarray(list(results))),np.divide(1,p_X))



    
 
