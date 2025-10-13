import torch
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from accuracy_bounds.inverseproblems.utils import insert_no_overlap_keep_A, sparse_block, offset_csr_block

def target_distances_cuda_V2(target_data, feasible_appartenance, p_X, batchsize):
    """
    Version 2: Computes pairwise distances between target samples using a feasible appartenance matrix that indicates traget data points belonging to one feasible set. 
    The computation is done batchwise on the nonzero entries of the common feasible appartenance matrix to avoid memory issues.

    Args:
        target_data: (DataLoader) Dataloader for target samples.
        feasible_appartenance: (torch.sparse.Tensor) feasible appartenance matrix.
        p_X: (int or float) Norm degree for distance computation.
        batchsize: (int) Batch size for processing.

    Returns:
        distsXX: (torch.sparse.Tensor) Sparse tensor of pairwise distances.

    """
    assert isinstance(target_data, DataLoader), 'Data input is only supported as Dataloader'
    #print(type(input_data.sampler), type(target_data1.sampler), type(target_data2.sampler))
    assert isinstance(target_data.sampler, SequentialSampler) \
           ,'Dataloaders with Random samplers are not supported'
    
    n_target = len(target_data.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    common_feasible = feasible_appartenance@(feasible_appartenance.transpose(0, 1))

    # First get the non zero indexes of common_feasible as a n,2 shaped array
    nonzero_indices = common_feasible.indices().numpy()
    index_array = np.vstack(nonzero_indices).T

    n_indexes = index_array.shape[0]

    if n_indexes %batchsize == 0:
        n_batches = n_indexes//batchsize
    else:
        n_batches = (n_indexes//batchsize)+1

    all_distances = []
    # Then Iterate batchwise manually over the nonzero indices
    for i_batch in range(n_batches):
        print(f"Target Batch: [{i_batch+1} / {n_batches}], ", end="\r")

        start_idx = i_batch * batchsize
        end_idx = min((i_batch+1)*batchsize, n_indexes)

        batch_indices = index_array[start_idx:end_idx, :]

        len_batch = batch_indices.shape[0]

        data1 = [target_data.dataset[batch_indices[i,0]] for i in range(len_batch)]
        data2 = [target_data.dataset[batch_indices[i,1]] for i in range(len_batch)]

        vectors1 = target_data.collate_fn(data1)
        vectors2 = target_data.collate_fn(data2)
        
        X1 = vectors1.reshape(vectors1.shape[0], -1)
        X2 = vectors2.reshape(vectors2.shape[0], -1)


        X1 = X1.to(device)
        X2 = X2.to(device)

        # Compute the norm of the difference for the batch
        distances_batch = torch.norm(X1-X2, p = p_X, dim = -1)
        # append to all_distances
        all_distances.append(distances_batch)

    #concatenate all the distances
    all_distances = torch.concatenate(all_distances)

    # Insert the distances into the 2d pairwise distances sparse tensor
    distsXX = torch.sparse_coo_tensor(indices=index_array.transpose(), values = all_distances, size=(n_target, n_target))

    return distsXX

def target_distances_batch_cuda(target_data1, target_data2 , p_X):
    """
    Computes pairwise distances between two batches of target data using the specified norm with order p_X.

    Args:
        target_data1: (torch.Tensor) batch of target data.
        target_data2: (torch.Tensor) batch of target data.
        p_X: (int or float) Norm degree for distance computation.

    Returns:
        torch.Tensor: Matrix of pairwise distances between target_data1 and target_data2.
    """
    x1_flat = target_data1.flatten(start_dim=1)
    x2_flat = target_data2.flatten(start_dim=1)

    #distancesXX = torch.norm(x1_flat[:,None, :]-x2_flat[None,:,:], p = p_X, dim = -1)
    distancesXX = torch.cdist(x1_flat, x2_flat, p = p_X)

    return distancesXX
    
def target_distances_dataloader_cuda(target_data1, target_data2, feasible_appartenance, p_X):
    """
    Computes pairwise distances between target samples in the same set using a feasible appartenance matrix
    allocating the sample indices to feasible sets.

    Args:
        target_data1: (DataLoader) Dataloader for target samples.
        target_data2: (DataLoader) Dataloader for target samples.
        feasible_appartenance: (torch.sparse.Tensor) Precomputed feasible appartenance matrix.
        p_X: (int or float) Norm degree for distance computation.
        batchsize: (int) Batch size for processing.

    Returns:
        distsXX: (torch.sparse.Tensor) Sparse tensor of pairwise distances.
    """
    assert isinstance(target_data1, DataLoader) and isinstance(target_data2, DataLoader) , 'Data input is only supported as Dataloader'

    assert isinstance(target_data1.sampler, SequentialSampler) and \
           isinstance(target_data2.sampler, SequentialSampler), \
           'Dataloaders with Random samplers are not supported'
    
    n_target = len(target_data1.dataset)

    common_feasible = feasible_appartenance@(feasible_appartenance.transpose(0, 1))

    distsXX = torch.sparse_coo_tensor(
        indices=torch.empty((2, 0), dtype=torch.long),  # no nonzero entries yet
        values=torch.tensor([], dtype=torch.float32),
        size=(n_target, n_target)
    )    

    for target_batch_id1, target_batch1 in enumerate(target_data1):

        for target_batch_id2, target_batch2 in enumerate(target_data2):
            print(f"Target Batch1: [{target_batch_id1+1} / {len(target_data1)}],     Target Batch2: [{target_batch_id2+1} / {len(target_data2)}]                    ", end="\r")

            batch_size_target = target_batch1.shape[0]

            idx_imin = batch_size_target * target_batch_id1
            idx_imax = min(idx_imin + batch_size_target, n_target)

            idx_jmin = batch_size_target * target_batch_id2
            idx_jmax = min(idx_jmin + batch_size_target, n_target)

            dists_small = target_distances_batch_cuda(target_batch1, target_batch2, p_X)
            
            common_feasible_small = sparse_block(common_feasible, idx_imin, idx_imax, idx_jmin, idx_jmax, out_layout="coo").to_dense()            
            mask = common_feasible_small != 0

            dists_small[~mask] = 0

            i0, i1 = idx_imin, idx_imax
            j0, j1 = idx_jmin, idx_jmax
            H, W = i1 - i0, j1 - j0

            dists_small = offset_csr_block(dists_small.to_sparse_csr(), i0, j0, (n_target, n_target))
            distsXX = insert_no_overlap_keep_A(distsXX, dists_small)

    return distsXX


def get_feasible_info(distsXX,feasible_appartenance):
    """
    Computes information for each feasible set F_y:
    - Diameter of F_y (maximum pairwise distance within F_y),
    - Indices of elements corresponding to the diameter,
    - Cardinality of F_y (number of elements in the feasible set).

    Args:
        distsXX: (scipy.sparse matrix) Pairwise distance matrix between target samples.
        feasible_appartenance: (scipy.sparse matrix) Feasible appartenance matrix.

    Returns:
        list: List of (diam_Fy, [i, j], cardinality) for each target sample.
    """

    def get_info(y_idx, fa, dXX):
        # Getting the indexes of x's in the F_y
        valid_idx = fa[:,y_idx].nonzero()[0]

        # restricting the distsXX matrix to F_y and doing the corresponding computations
        subdistXX = dXX[valid_idx, :][:, valid_idx]
        subdistXX = subdistXX.toarray() 
        
        if subdistXX.size == 0:
            return 0, (None, None), 0
        

        diam_Fy = np.nanmax(subdistXX)
        flat_index = np.nanargmax(subdistXX)
        row, col = np.unravel_index(flat_index, subdistXX.shape)

        i = valid_idx[row]
        j = valid_idx[col]
        if False:
            print(f'y space : {y_idx}. Diam in {i}, {j}')

        return float(diam_Fy), [int(i), int(j)], int(subdistXX.shape[0])
    n,p = feasible_appartenance.shape

    return list(Parallel(n_jobs=-1, backend='threading')(delayed(get_info)(y_idx, feasible_appartenance, distsXX) for y_idx in tqdm(range(p))))


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



    
 
