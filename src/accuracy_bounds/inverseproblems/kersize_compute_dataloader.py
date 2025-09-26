import torch
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
import time
from torch.utils.data import DataLoader, SequentialSampler
from scipy import sparse



def norms_batch_cuda(target_null_data, p_X):
    x_flat = target_null_data.flatten(start_dim = 1)
    return torch.norm(x_flat, p = p_X, dim=-1)

def norms_perbatch_cuda(target_null_data, p_X):
    assert isinstance(target_null_data, DataLoader)
    assert isinstance(target_null_data.sampler, SequentialSampler)

    n_target_null = len(target_null_data.dataset)

    norms_all = torch.zeros(n_target_null)
    for target_null_batch_id,target_null_batch in enumerate(target_null_data):
        print(f"Target Batch: [{target_null_batch_id+1} / {len(target_null_data)}]                    ", end="\r")

        norms_small = norms_batch_cuda(target_null_batch['img'], p_X=p_X)
        batch_size_current = target_null_batch['img'].shape[0]

        idx_imin = batch_size_current * target_null_batch_id
        idx_imax = batch_size_current * (target_null_batch_id + 1)

        norms_all[idx_imin:idx_imax] = norms_small

    return norms_all

def avgLB_sym_samplingYX(null_norms, p):
    return np.nanmean(null_norms**p)**(1/p)



def insert_no_overlap_keep_A(A_coo, B_csr):
    A = A_coo.coalesce()
    B = B_csr.to_sparse_coo()
    I_A, V_A = A.indices(), A.values()
    I_B, V_B = B.indices(), B.values()
    # linearize indices to detect overlaps
    lin_A = I_A[0] * A.shape[1] + I_A[1]
    lin_B = I_B[0] * A.shape[1] + I_B[1]
    # keep only B entries not in A
    mask = ~torch.isin(lin_B, lin_A)
    I = torch.cat([I_A, I_B[:, mask]], dim=1)
    V = torch.cat([V_A, V_B[mask]], dim=0)
    return torch.sparse_coo_tensor(I, V, A.shape, device=A.device, dtype=A.dtype)

def sparse_block(A, i0, i1, j0, j1, out_layout="coo"):
    """
    Return A[i0:i1, j0:j1] as a sparse tensor without densifying.
    Works for A in COO or CSR. out_layout: 'coo' or 'csr'
    """
    Acoo = A.to_sparse_coo().coalesce()
    r, c = Acoo.indices()
    v     = Acoo.values()

    m = (r >= i0) & (r < i1) & (c >= j0) & (c < j1)
    r2 = r[m] - i0
    c2 = c[m] - j0
    v2 = v[m]

    B = torch.sparse_coo_tensor(
        torch.stack([r2, c2], dim=0),
        v2,
        (i1 - i0, j1 - j0),
        device=A.device, dtype=A.dtype
    ).coalesce()

    return B if out_layout == "coo" else B.to_sparse_csr()

#TODO : allow any custom norm in the distance computation (not only the normm p)

def target_distances_samplingYX_precomputedFA_cuda_V2(A, target_data, feasible_appartenance, p_X, batchsize):
    '''
    Same as target_distances_samplingYX_perbatch_cuda but here, the feasible appartenance matrix is precomputed
    '''
    assert isinstance(target_data, DataLoader), 'Data input is only supported as Dataloader'
    #print(type(input_data.sampler), type(target_data1.sampler), type(target_data2.sampler))
    assert isinstance(target_data.sampler, SequentialSampler) \
           ,'Dataloaders with Random samplers are not supported'
    
    n_target = len(target_data.dataset)
  

    common_feasible = feasible_appartenance@(feasible_appartenance.transpose(0, 1))

    # First get the non zero indexes of common_feasible as a n,2 shaped array
    from pdb import set_trace
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

        # Load in X1 the vectors corresponding to the left index, and in X2 for the right index
        len_batch = batch_indices.shape[0]
        X1, X2 = [], []
        for i in range(len_batch):
            X1.append(target_data.dataset[batch_indices[i,0]]['img'])
            X2.append(target_data.dataset[batch_indices[i,1]]['img'])
        X1 = torch.stack(X1)
        X2 = torch.stack(X2)
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

            batch_size_target = target_batch1["img"].shape[0]

            idx_imin = batch_size_target * target_batch_id1
            idx_imax = batch_size_target * (target_batch_id1 + 1)

            idx_jmin = batch_size_target * target_batch_id2
            idx_jmax = batch_size_target * (target_batch_id2 + 1)

            dists_small = distsXX_samplingYX_batch_cuda(A, target_batch1["img"], target_batch2["img"], p_X)
            
            common_feasible_small = sparse_block(common_feasible, idx_imin, idx_imax, idx_jmin, idx_jmax, out_layout="coo").to_dense()            
            mask = common_feasible_small != 0

            dists_small[~mask] = 0

            i0, i1 = idx_imin, idx_imax
            j0, j1 = idx_jmin, idx_jmax
            H, W = i1 - i0, j1 - j0

            dists_small = offset_csr_block(dists_small.to_sparse_csr(), i0, j0, (n_target, n_target))
            distsXX = insert_no_overlap_keep_A(distsXX, dists_small)
    return distsXX, feasible_appartenance
    

def target_distances_samplingYX_perbatch_cuda(A, input_data, target_data1, target_data2, forwarded_target, p_X, p_Y, epsilon):
    """
    Computes pairwise distances between target samples (in X), considering only those pairs that belong to the same feasible set F_y.
    Splits data into batches for efficient computation and constructs a matrix indicating which x belongs to which F_y.

    Args:
        input_data (torch.ndarray): Input data samples (y).
        target_data (torch.ndarray): Target data samples (x).
        forwarded_target (torch.ndarray): Forwarded target data (F(x,e)).
        p_X (int or float): Norm degree for target distance (X) computation.
        p_Y (int or float): Norm degree for feasibility computation.
        epsilon (float): Noise level in the model.
        batch_size (int): Size of each batch for computation.

    Returns:
        distsXX (scipy.sparse.lil_matrix): Matrix of pairwise distances between feasible target samples.
        feasible_appartenance (scipy.sparse.csr_matrix): Feasibility matrix indicating which x belongs to which F_y.
    """
    assert isinstance(input_data, DataLoader) and isinstance(target_data1, DataLoader) and isinstance(target_data2, DataLoader) and isinstance(forwarded_target, DataLoader) , 'Data input is only supported as Dataloader'
    #print(type(input_data.sampler), type(target_data1.sampler), type(target_data2.sampler))
    assert isinstance(input_data.sampler, SequentialSampler) and \
           isinstance(target_data1.sampler, SequentialSampler) and \
           isinstance(forwarded_target, DataLoader) and \
           isinstance(target_data2.sampler, SequentialSampler), \
           'Dataloaders with Random samplers are not supported'

    n_input = len(input_data.dataset)
    n_target = len(forwarded_target.dataset)

    feasible_appartenance = torch.sparse_coo_tensor(indices=torch.empty((2, 0), dtype=torch.long),  # no nonzero entries yet
                            values=torch.tensor([], dtype=torch.float32),
                            size=(n_target, n_input))
    

    for target_batch_id, target_batch in enumerate(forwarded_target):

        for input_batch_id, input_batch in enumerate(input_data):
            print(f"Target Batch: [{target_batch_id+1} / {len(forwarded_target)}],     Input Batch: [{input_batch_id+1} / {len(input_data)}]                    ", end="\r")
                
            feasible_small = feasibleApp_samplingYX_batch_cuda(A, input_batch["img"], target_batch["img"], p_Y, epsilon)
            feasible_small = feasible_small.to_sparse_csr()

            batch_size_target = target_batch["img"].shape[0]
            batch_size_input = input_batch["img"].shape[0]

            # TODO: what if batchsize is not always the same? e.g. drop_last = False
            idx_imin = batch_size_target * target_batch_id
            idx_imax = batch_size_target * (target_batch_id + 1)

            idx_jmin = batch_size_input * input_batch_id
            idx_jmax = batch_size_input * (input_batch_id + 1)

            i0, i1 = idx_imin, idx_imax
            j0, j1 = idx_jmin, idx_jmax

            feasible_small = offset_csr_block(feasible_small.to_sparse_csr(), i0, j0, (n_target, n_target))

            feasible_appartenance = insert_no_overlap_keep_A(feasible_appartenance, feasible_small.detach().cpu())

    #print(feasible_appartenance.data.nbytes/(1024*1024), f'size data {feasible_appartenance.shape}') 
    feasible_appartenance = feasible_appartenance.to_sparse_csr()

    common_feasible = feasible_appartenance@(feasible_appartenance.transpose(0, 1))

    distsXX = torch.sparse_coo_tensor(
        indices=torch.empty((2, 0), dtype=torch.long),  # no nonzero entries yet
        values=torch.tensor([], dtype=torch.float32),
        size=(n_target, n_target)
    )    

    for target_batch_id1, target_batch1 in enumerate(target_data1):

        for target_batch_id2, target_batch2 in enumerate(target_data2):
            print(f"Target Batch1: [{target_batch_id1+1} / {len(target_data1)}],     Target Batch2: [{target_batch_id2+1} / {len(target_data2)}]                    ", end="\r")

            batch_size_target = target_batch1["img"].shape[0]

            idx_imin = batch_size_target * target_batch_id1
            idx_imax = batch_size_target * (target_batch_id1 + 1)

            idx_jmin = batch_size_target * target_batch_id2
            idx_jmax = batch_size_target * (target_batch_id2 + 1)

            dists_small = distsXX_samplingYX_batch_cuda(A, target_batch1["img"], target_batch2["img"], p_X)
            
            common_feasible_small = sparse_block(common_feasible, idx_imin, idx_imax, idx_jmin, idx_jmax, out_layout="coo").to_dense()            
            mask = common_feasible_small != 0

            dists_small[~mask] = 0

            i0, i1 = idx_imin, idx_imax
            j0, j1 = idx_jmin, idx_jmax
            H, W = i1 - i0, j1 - j0

            dists_small = offset_csr_block(dists_small.to_sparse_csr(), i0, j0, (n_target, n_target))
            distsXX = insert_no_overlap_keep_A(distsXX, dists_small)
    return distsXX, feasible_appartenance

def offset_csr_block(local_csr: torch.Tensor, i0: int, j0: int, global_shape):
    """
    Place a CSR 'local_csr' block at (row=i0, col=j0) inside a larger CSR tensor of shape 'global_shape'.

    local_csr: torch.sparse_csr_tensor of shape (h, w)
    i0, j0:    top-left offsets (rows, cols) in the global matrix
    global_shape: (H, W)
    returns: torch.sparse_csr_tensor of shape (H, W)
    """
    assert local_csr.layout == torch.sparse_csr, "local_csr must be CSR"
    H, W = global_shape
    h, w = local_csr.shape
    if not (0 <= i0 <= H - h):
        raise ValueError(f"Row offset i0={i0} with block height {h} exceeds H={H}")
    if not (0 <= j0 <= W - w):
        raise ValueError(f"Col offset j0={j0} with block width {w} exceeds W={W}")

    crow = local_csr.crow_indices()
    col  = local_csr.col_indices()
    val  = local_csr.values()

    # shift columns
    col_off = col + j0

    nnz = val.numel()
    device = local_csr.device
    itype  = crow.dtype

    # Build global crow: zeros up to i0, then local crow, then constant nnz afterwards
    crow_g = torch.empty(H + 1, dtype=itype, device=device)
    if i0 > 0:
        crow_g[:i0] = 0
    crow_g[i0:i0 + h + 1] = crow
    if i0 + h < H:
        crow_g[i0 + h + 1:] = nnz

    return torch.sparse_csr_tensor(crow_g, col_off, val, size=global_shape,
                                   device=device, dtype=local_csr.dtype)



# TODO: Why is A needed here? 
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

def feasibleApp_samplingYX_perbatch_cuda(A, input_data, forwarded_target, p_Y, epsilon):
    assert isinstance(input_data, DataLoader) and isinstance(forwarded_target, DataLoader) , 'Data input is only supported as Dataloader'
    #print(type(input_data.sampler), type(target_data1.sampler), type(target_data2.sampler))
    assert isinstance(input_data.sampler, SequentialSampler) and \
           isinstance(forwarded_target, DataLoader) and \
           'Dataloaders with Random samplers are not supported'

    n_input = len(input_data.dataset)
    n_target = len(forwarded_target.dataset)

    feasible_appartenance = torch.sparse_coo_tensor(indices=torch.empty((2, 0), dtype=torch.long),  # no nonzero entries yet
                            values=torch.tensor([], dtype=torch.float32),
                            size=(n_target, n_input))
    

    for target_batch_id, target_batch in enumerate(forwarded_target):

        for input_batch_id, input_batch in enumerate(input_data):
            print(f"Target Batch: [{target_batch_id+1} / {len(forwarded_target)}],     Input Batch: [{input_batch_id+1} / {len(input_data)}]                    ", end="\r")
                
            feasible_small = feasibleApp_samplingYX_batch_cuda(A, input_batch["img"], target_batch["img"], p_Y, epsilon)
            feasible_small = feasible_small.to_sparse_csr()

            batch_size_target = target_batch["img"].shape[0]
            batch_size_input = input_batch["img"].shape[0]

            # TODO: what if batchsize is not always the same? e.g. drop_last = False
            idx_imin = batch_size_target * target_batch_id
            idx_imax = batch_size_target * (target_batch_id + 1)

            idx_jmin = batch_size_input * input_batch_id
            idx_jmax = batch_size_input * (input_batch_id + 1)

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

    #e_diff = -forwarded_flat[:,None, :] + input_data_flat[None,:,:]

    #feasible_appartenance = torch.norm(e_diff,p = p_Y, dim = -1)<epsilon

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

def avgkersize_samplingYX(distsXX, feasible_appartenance, p_X):
    """
    Computes the average kernel size approcimation.

    Args:
        distsXX (scipy.sparse matrix): Pairwise distance matrix between target samples.
        feasible_appartenance (torch.ndarray or sparse matrix): Feasibility matrix.
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
    results = Parallel(n_jobs=-1)(delayed(compute_mean_distance)(y_idx, feasible_appartenance, distsXX) for y_idx in tqdm(range(p)))

 
    return np.nanmean(np.asarray(results))**(1/p_X)



def target_distances_samplingX_batch_cuda(A,input_data, target_data, p_X, p_Y, epsilon):
    """
    Computes pairwise distances between target samples, masking out pairs whose corresponding inputs are not in some common feasible set.

    Args:
        input_data (torch.ndarray): Input data samples.
        target_data (torch.ndarray): Target data samples.
        p_X (int or float): Norm degree for target distance computation.
        p_Y (int or float): Norm degree for feasibility computation.
        epsilon (float): Feasibility threshold.

    Returns:
        torch.ndarray: Masked pairwise distance matrix for target samples.
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
        torch.ndarray: Masked pairwise distance matrix for target samples.
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
        input_data (torch.ndarray): Input data samples.
        target_data (torch.ndarray): Target data samples.
        p_X (int or float): Norm degree for target distance computation.
        p_Y (int or float): Norm degree for feasibility computation.
        epsilon (float): Feasibility threshold.
        batch_size (int): Size of each batch for computation.

    Returns:
        torch.ndarray: Masked pairwise distance matrix for target samples.
        torch.ndarray: Boolean matrix indicating feasibility between input samples.
    """
    n = target_data.shape[0]
    n_batches = (n//batch_size)

    masked_target_dists = torch.zeros((n,n))
    same_feasible = torch.zeros((n,n))

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
        masked_target_dists (torch.ndarray): Masked pairwise distance matrix for target samples.

    Returns:
        float: Kersize.
    """
    return np.nanmax(masked_target_dists)


if __name__ =='__main__':

    class RandomDataset(torch.utils.data.Dataset):
        def __init__(self, data_array, suffix):
            if suffix == 'x':
                self.data = data_array
            elif suffix == 'y':
                self.data = data_array[:,:2]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return {'idx': idx, 'img': self.data[idx]}
        

    # Parameters
    n = 10000  # Number of vectors in dataset
    epsilon = 0.039  # Distance threshold for similarity
    batch_size = 1000  # Batch size for DataLoader

    data_3D = torch.randn(n, 3).float()
    data_3D = data_3D / data_3D.norm(dim=1, keepdim=True) # Random points on the unit sphere

    # Create datasets and dataloaders for dataset1 and dataset2
    dataset_3D = RandomDataset(data_array= data_3D,suffix='x')
    dataset_proj = RandomDataset(data_array=data_3D, suffix='y')


    dataloader_3D = DataLoader(dataset_3D, batch_size=batch_size, shuffle=False)
    dataloader_3D_2 = DataLoader(dataset_3D, batch_size=batch_size, shuffle=False)
    dataloader_proj = DataLoader(dataset_proj, batch_size=batch_size, shuffle=False)
    dataloader_forwarded3D = DataLoader(dataset_proj, batch_size=batch_size, shuffle=False)

    feas_app = feasibleApp_samplingYX_perbatch_cuda(0, dataloader_proj, dataloader_forwarded3D, p_Y=2, epsilon=epsilon)
    t0 = time.time()
    distsXX_new, feas_app_new = target_distances_samplingYX_precomputedFA_cuda_V2(0, dataloader_3D, feas_app, p_X = 1, batchsize=10000)
    t1 = time.time()
    distsXX_old, feas_app_old = target_distances_samplingYX_precomputedFA_perbatch_cuda(0,dataloader_3D, dataloader_3D_2, feas_app, p_X = 1)
    t2 = time.time()
    print(f'Time taken for the new method= {np.round(t1-t0, 3)} seconds')
    print(f'Time taken for the old method= {np.round(t2-t1, 3)} seconds')

    h,w = feas_app.shape
    print(f'Sparsity ratio of feasible apprtenance = {feas_app.sum()/(h*w)}')
    print(torch.norm(distsXX_old.to_dense())**2)  
    print(torch.norm(distsXX_new.to_dense())**2)  

    from pdb import set_trace
    set_trace()
    
 
