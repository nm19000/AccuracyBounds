import torch
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from scipy import sparse
from accuracy_bounds.inverseproblems.utils import insert_no_overlap_keep_A, sparse_block, offset_csr_block

def feasible_appartenance_additive_noise_batch_cuda(input_data, forwarded_target, p_Y, epsilon):
    """
    Computes the feasible appartenance matrix for one batch, where feasible sets are saved over pairwise indices in a matrix. 
    The data samples are allocated for a forward model with additive noise by 
    filtering candidates using an upper bound on the norm difference with the noise level epsilon.
    Efficiency depends on the data structure. 

    Args:
        input_data: (DataLoader) Dataloader for approximate inverse map input samples.
        forwarded_target: (DataLoader) Dataloader for forwarded target samples or for approximate inverse map input samples.
        p_Y: (int or float) Norm degree on measurment data space.
        epsilon: (float) Noise level of additive noise from the forward model.


    Returns:
        torch.sparse.Tensor: (Boolean matrix) Feasible appartenance matrix.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forwarded_target = torch.tensor(forwarded_target, dtype = torch.float32, device = device)
    input_data = torch.tensor(input_data, dtype = torch.float32, device = device)

    n = input_data.shape[0]
    m = forwarded_target.shape[0]
    input_data_flat = torch.tensor(input_data.reshape(n,-1), device = device)
    forwarded_flat = torch.tensor(forwarded_target.reshape(m,-1),device = device)

    feasible_appartenance = torch.cdist(forwarded_flat, input_data_flat, p = p_Y)<epsilon


    return feasible_appartenance 

def feasible_appartenance_additive_noise_dataloader_cuda(input_data, forwarded_target, p_Y, epsilon):
    """
    Computes the feasible appartenance matrix by evaluating all pairs in batches.

    Args:
        input_data (DataLoader): Dataloader for input samples.
        forwarded_target (DataLoader): Dataloader for forwarded target samples.
        p_Y (int or float): Norm degree for feasibility computation.
        epsilon (float): Feasibility threshold.

    Returns:
        torch.sparse.Tensor: Feasible appartenance matrix.
    """
    assert isinstance(input_data, DataLoader) and isinstance(forwarded_target, DataLoader) , 'Data input is only supported as Dataloader'

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
                
            feasible_small = feasible_appartenance_additive_noise_batch_cuda(input_batch, target_batch, p_Y, epsilon)
            feasible_small = feasible_small.to_sparse_csr()

            batch_size_target = target_batch.shape[0]
            batch_size_input = input_batch.shape[0]

            idx_imin = batch_size_target * target_batch_id
            idx_imax = min(idx_imin + batch_size_target, n_target)

            idx_jmin = batch_size_input * input_batch_id
            idx_jmax = min(idx_jmin + batch_size_input, n_input)

            i0, i1 = idx_imin, idx_imax
            j0, j1 = idx_jmin, idx_jmax

            feasible_small = offset_csr_block(feasible_small.to_sparse_csr(), i0, j0, (n_target, n_target))

            feasible_appartenance = insert_no_overlap_keep_A(feasible_appartenance, feasible_small.detach().cpu())


    feasible_appartenance = feasible_appartenance.to_sparse_coo()
    return feasible_appartenance


def feasible_appartenance_additive_noise_cuda(input_data, forwarded_target, p_Y, epsilon, batchsize):
    """
    Second (possibly faster) way to compute the feasible sets by an appartenance matric.
    Computes the feasible appartenance matrix, where feasible sets are saved over pairwise indices in a matrix. 
    The data samples are allocated for a forward model with additive noise by 
    filtering candidates using an upper bound on the norm difference with the noise level epsilon.
    Efficiency depends on the data structure. 

    Args:
        input_data: (DataLoader) Dataloader for approximate inverse map input samples.
        forwarded_target: (DataLoader) Dataloader for forwarded target samples or for approximate inverse map input samples.
        p_Y: (int or float) Norm degree on measurment data space.
        epsilon: (float) Noise level of additive noise from the forward model.
        batchsize: (int) Batch size for processing.

    Returns:
        torch.sparse.Tensor: Feasible appartenance matrix.
    """
    assert isinstance(input_data, DataLoader) and isinstance(forwarded_target, DataLoader) , 'Data input is only supported as Dataloader'
    #print(type(input_data.sampler), type(target_data1.sampler), type(target_data2.sampler))
    assert isinstance(input_data.sampler, SequentialSampler) and \
           isinstance(forwarded_target.sampler, SequentialSampler) and \
           'Dataloaders with Random samplers are not supported'

    n_input = len(input_data.dataset)
    n_target = len(forwarded_target.dataset)

    feasible_appartenance_candidates = sparse.csr_matrix((n_target, n_input), dtype=int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # First compute the LB of the norm diff
    #   Distinguish L2 and others for the LB
    #   If L2, get the mean and variance of each datapoint and then compute the pairwise Lb of the diff norm
    #   Else, compute the norm of every datapoint and use reverse triangle inequality
    
    norms_input = []
    for input_batch_id, input_batch in enumerate(input_data):
        n_batches = len(input_data)
        print(f"Input Batch: [{input_batch_id+1} / {n_batches}], ", end="\r")

        input_batch = input_batch.to(device)

        norms_batch = torch.norm(input_batch.reshape(input_batch.shape[0], -1), p = p_Y, dim=1)
        norms_input.append(norms_batch)

    norms_input = torch.concatenate(norms_input)

    for target_batch_id, target_batch in enumerate(forwarded_target):
        
        n_batches = len(forwarded_target)
        print(f"Input Batch: [{target_batch_id+1} / {n_batches}], ", end="\r")

        target_batch = target_batch.to(device)

        norms_batch = torch.norm(target_batch.reshape(target_batch.shape[0], -1), p = p_Y, dim=1)

        # Compute the pairwise absolute difference of the norms
        feas_candidates_small = torch.abs(norms_batch[:,None]- norms_input[None,:])<epsilon

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

    all_feas_app = []
    # Then Iterate batchwise manually over the nonzero indices
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
        X2 = vectors2.reshape(vectors2.shape[0], -1)


        X1 = X1.to(device)
        X2 = X2.to(device)

        # Compute the norm of the difference for the batch
        feas_app_batch = torch.norm(X1-X2, p = p_Y, dim = -1)<epsilon
        all_feas_app.append(feas_app_batch)
    all_feas_app = torch.concatenate(all_feas_app)

    feasible_appartenance = torch.sparse_coo_tensor(indices=index_array.transpose(), values = all_feas_app, size=(n_target, n_input))
        
    return feasible_appartenance




