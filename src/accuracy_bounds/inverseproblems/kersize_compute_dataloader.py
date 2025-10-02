import torch
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
import time
from torch.utils.data import DataLoader, SequentialSampler
from scipy import sparse
from accuracy_bounds.inverseproblems.utils import insert_no_overlap_keep_A, sparse_block, offset_csr_block


# TODO: Why is A needed here? 

def target_distances_samplingYX_precomputedFA_cuda_V2(A, target_data, feasible_appartenance, p_X, batchsize):
    '''
    Same as target_distances_samplingYX_perbatch_cuda but here, the feasible appartenance matrix is precomputed
    '''
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

    return distsXX, feasible_appartenance

    
def target_distances_samplingYX_precomputedFA_perbatch_cuda(A, target_data1, target_data2, feasible_appartenance, p_X):
    '''
    Same as target_distances_samplingYX_perbatch_cuda but here, the feasible appartenance matrix is precomputed
    '''
    assert isinstance(target_data1, DataLoader) and isinstance(target_data2, DataLoader) , 'Data input is only supported as Dataloader'
    #print(type(input_data.sampler), type(target_data1.sampler), type(target_data2.sampler))
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

            dists_small = distsXX_samplingYX_batch_cuda(A, target_batch1, target_batch2, p_X)
            
            common_feasible_small = sparse_block(common_feasible, idx_imin, idx_imax, idx_jmin, idx_jmax, out_layout="coo").to_dense()            
            mask = common_feasible_small != 0

            dists_small[~mask] = 0

            i0, i1 = idx_imin, idx_imax
            j0, j1 = idx_jmin, idx_jmax
            H, W = i1 - i0, j1 - j0

            dists_small = offset_csr_block(dists_small.to_sparse_csr(), i0, j0, (n_target, n_target))
            distsXX = insert_no_overlap_keep_A(distsXX, dists_small)
    return distsXX, feasible_appartenance

def distsXX_samplingYX_batch_cuda(A, target_data1, target_data2 , p_X):
    """
    Computes pairwise distances between two batches of target data (x) using the specified norm.

    Args:
        target_data1 (torch.ndarray): First batch of target data.
        target_data2 (torch.ndarray): Second batch of target data.
        p_X (int or float): Norm degree for distance computation.

    Returns:
        torch.Tensor: Matrix of pairwise distances between target_data1 and target_data2.
    """
    x1_flat = target_data1.flatten(start_dim=1)
    x2_flat = target_data2.flatten(start_dim=1)

    #distancesXX = torch.norm(x1_flat[:,None, :]-x2_flat[None,:,:], p = p_X, dim = -1)
    distancesXX = torch.cdist(x1_flat, x2_flat, p = p_X)

    return distancesXX



def feasibleApp_samplingYX_linear_cuda(A, input_data, forwarded_target, p_Y, epsilon, batchsize):
    assert isinstance(input_data, DataLoader) and isinstance(forwarded_target, DataLoader) , 'Data input is only supported as Dataloader'
    #print(type(input_data.sampler), type(target_data1.sampler), type(target_data2.sampler))
    assert isinstance(input_data.sampler, SequentialSampler) and \
           isinstance(forwarded_target.sampler, SequentialSampler) and \
           'Dataloaders with Random samplers are not supported'
    '''
    It Avoids double loop into the dataset by first computing a lower bound of the distance between y_i and y_j using triangle inequality
    Then, it computes exactly the distance only for the pairs (i,j) where the lower
    '''
    n_input = len(input_data.dataset)
    n_target = len(forwarded_target.dataset)

    feasible_appartenance_candidates = sparse.csr_matrix((n_target, n_input), dtype=int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # First compute the LB of the norm diff
    #   Distinguish L2 and others for the LB
    #   If L2, get the mean and variance of each datapoint and then compute the pairwise Lb of the diff norm
    #   Else, compute the norm of every datapoint and use reverse triangle inequality
    
    print('Filtering candidates with distance LB')
    norms_input = []
    print('Loading input batches')
    for input_batch_id, input_batch in enumerate(input_data):
        n_batches = len(input_data)
        print(f"Input Batch: [{input_batch_id+1} / {n_batches}], ", end="\r")

        input_batch = input_batch.to(device)

        norms_batch = torch.norm(input_batch.reshape(input_batch.shape[0], -1), p = p_Y, dim=1)
        norms_input.append(norms_batch)

    norms_input = torch.concatenate(norms_input)

    print('Loading forwarded target batches')
    for target_batch_id, target_batch in enumerate(forwarded_target):
        
        n_batches = len(forwarded_target)
        print(f"Input Batch: [{target_batch_id+1} / {n_batches}], ", end="\r")

        target_batch = target_batch.to(device)

        norms_batch = torch.norm(target_batch.reshape(target_batch.shape[0], -1), p = p_Y, dim=1)

        # Compute the pairwise absolute difference of the norms
        feas_candidates_small = torch.abs(norms_batch[:,None]- norms_input[None,:])<epsilon


        if target_batch_id %50 ==35 and False:
            h,w = feasible_appartenance_candidates.shape
            print(f'Sparsity ratio of feas_candidates : {feasible_appartenance_candidates.nnz/(idx_imin*w)}')
            print(f'Sizes : \n Norms batch : {norms_batch.nbytes/(1024*1024)} MB \n feas_candidates_small : {feas_candidates_small.nbytes/(1024*1024)} MB \n feasible_appartenance_candidates : {feasible_appartenance_candidates.data.nbytes/(1024*1024)} MB')
            
        batch_size_target = target_batch.shape[0]

        idx_imin = batch_size_target * target_batch_id
        idx_imax = min(idx_imin+ batch_size_target,n_target )

        feasible_appartenance_candidates[idx_imin:idx_imax, :] = sparse.csr_matrix(feas_candidates_small.cpu().numpy())

        del norms_batch
        del feas_candidates_small

    # Retrieve all the cndidate indices where the dist can be lower than epsilon
    feasible_appartenance_candidates = feasible_appartenance_candidates.tocoo()
    # First get the non zero indexes of common_feasible as a n,2 shaped array
    nonzero_indices = np.vstack((feasible_appartenance_candidates.row, feasible_appartenance_candidates.col))
    index_array = nonzero_indices.T

    n_indexes = index_array.shape[0]
    # Compute the exact diff on those candidates

    if n_indexes %batchsize == 0:
        n_batches = n_indexes//batchsize
    else:
        n_batches = (n_indexes//batchsize)+1
    print(f'Number of candidates : {n_indexes} \n ratio = {n_indexes/(n_input*n_target)}')

    all_feas_app = []
    # Then Iterate batchwise manually over the nonzero indices
    print('Computing exact distances for candidates')
    for i_batch in range(n_batches):
        
        print(f" Batch: [{i_batch+1} / {n_batches}], ", end="\r")

        start_idx = i_batch * batchsize
        end_idx = min((i_batch+1)*batchsize, n_indexes)

        batch_indices = index_array[start_idx:end_idx, :]

        # Load in X1 the vectors corresponding to the left index, and in X2 for the right index
        len_batch = batch_indices.shape[0]

        data1 = [forwarded_target.dataset[batch_indices[i,0]] for i in range(len_batch)]
        data2 = [forwarded_target.dataset[batch_indices[i,1]] for i in range(len_batch)]

        vectors1 = forwarded_target.collate_fn(data1)
        vectors2 = forwarded_target.collate_fn(data2)
        
        X1 = vectors1.reshape(vectors1.shape[0], -1)
        X2 = vectors2.reshape(vectors2.shape[0, -1])


        X1 = X1.to(device)
        X2 = X2.to(device)

        # Compute the norm of the difference for the batch
        feas_app_batch = torch.norm(X1-X2, p = p_Y, dim = -1)<epsilon
        all_feas_app.append(feas_app_batch)
    all_feas_app = torch.concatenate(all_feas_app)

    feasible_appartenance = torch.sparse_coo_tensor(indices=index_array.transpose(), values = all_feas_app, size=(n_target, n_input))
        
    return feasible_appartenance

def feasibleApp_samplingYX_perbatch_cuda(A, input_data, forwarded_target, p_Y, epsilon):
    assert isinstance(input_data, DataLoader) and isinstance(forwarded_target, DataLoader) , 'Data input is only supported as Dataloader'
    #print(type(input_data.sampler), type(target_data1.sampler), type(target_data2.sampler))
    assert isinstance(input_data.sampler, SequentialSampler) and \
           isinstance(forwarded_target.sampler, SequentialSampler) and \
           'Dataloaders with Random samplers are not supported'

    n_input = len(input_data.dataset)
    n_target = len(forwarded_target.dataset)

    feasible_appartenance = torch.sparse_coo_tensor(indices=torch.empty((2, 0), dtype=torch.long),  # no nonzero entries yet
                            values=torch.tensor([], dtype=torch.float32),
                            size=(n_target, n_input))
    

    for target_batch_id, target_batch in enumerate(forwarded_target):

        for input_batch_id, input_batch in enumerate(input_data):
            print(f"Target Batch: [{target_batch_id+1} / {len(forwarded_target)}],     Input Batch: [{input_batch_id+1} / {len(input_data)}]                    ", end="\r")
                
            feasible_small = feasibleApp_samplingYX_batch_cuda(A, input_batch, target_batch, p_Y, epsilon)
            feasible_small = feasible_small.to_sparse_csr()

            batch_size_target = target_batch.shape[0]
            batch_size_input = input_batch.shape[0]

            # TODO: what if batchsize is not always the same? e.g. drop_last = False
            idx_imin = batch_size_target * target_batch_id
            idx_imax = min(idx_imin + batch_size_target, n_target)

            idx_jmin = batch_size_input * input_batch_id
            idx_jmax = min(idx_jmin + batch_size_input, n_input)

            i0, i1 = idx_imin, idx_imax
            j0, j1 = idx_jmin, idx_jmax

            feasible_small = offset_csr_block(feasible_small.to_sparse_csr(), i0, j0, (n_target, n_target))

            feasible_appartenance = insert_no_overlap_keep_A(feasible_appartenance, feasible_small.detach().cpu())

    #print(feasible_appartenance.data.nbytes/(1024*1024), f'size data {feasible_appartenance.shape}') 
    feasible_appartenance = feasible_appartenance.to_sparse_coo()
    return feasible_appartenance

def feasibleApp_samplingYX_batch_cuda(A, input_data, forwarded_target, p_Y, epsilon):
    """
    Determines which target samples belong to the feasible set of each input data sample.

    Args:
        input_data (torch.ndarray): Batch of input samples (y).
        forwarded_target (torch.ndarray): Batch of forwarded target samples (F(x,e)).
        p_Y (int or float): Norm degree for feasibility computation.
        epsilon (float): Feasibility threshold.

    Returns:
        torch.ndarray: Boolean matrix indicating feasibility (shape: [forwarded_target, input_data]).
    """
    from pdb import set_trace
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forwarded_target = torch.tensor(forwarded_target, dtype = torch.float32, device = device)
    input_data = torch.tensor(input_data, dtype = torch.float32, device = device)

    n = input_data.shape[0]
    m = forwarded_target.shape[0]
    input_data_flat = torch.tensor(input_data.reshape(n,-1), device = device)
    forwarded_flat = torch.tensor(forwarded_target.reshape(m,-1),device = device)

    feasible_appartenance = torch.cdist(forwarded_flat, input_data_flat, p = p_Y)<epsilon


    return feasible_appartenance # feasible appartenance is y vs x


def get_feasible_info(distsXX,feasible_appartenance):
    # Get the information of each F_y : diam(F_y), the indexes of elements corresponding to diam(F_y), the cardinal of F_y

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

def avgLB_samplingYX(distsXX, feasible_appartenance, p):
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
        return np.nanmean(subdistXX**p)



    n,p = feasible_appartenance.shape
    results = Parallel(n_jobs=-1)(delayed(compute_mean_distance)(y_idx, feasible_appartenance, distsXX) for y_idx in tqdm(range(p)))

    return np.nanmean(np.asarray(list(results)))**(1/p)



    
 
