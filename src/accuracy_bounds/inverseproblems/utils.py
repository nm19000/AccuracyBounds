import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
import torch
import scipy.sparse as sp



#To calculate operators in the form of matrices (suits better to big operators)
class MatrixOpCalculator:
    def __init__(self, n_in, n_out, Operator, num_workers = None, singular_threshold_ratio = 0.001):
        self.n_in = n_in
        self.n_out = n_out
        self.operator = Operator
        self.num_workers=num_workers
        if num_workers is None:
            self.num_workers = cpu_count()
        self.singular_threshold_ratio = singular_threshold_ratio
    
    # Compute column i of the sparse matrix
    def compute_column(self,i):
        A = self.operator
        e_i = np.zeros(self.n_in)
        e_i[i] = 1.0
        col = A(e_i)

        # Find non-zero entries
        row_idx = np.nonzero(col)[0]
        data = col[row_idx]
        col_idx = np.full_like(row_idx, i)
      
        return row_idx, col_idx, data
    
    def build_sparse_matrix_parallel(self):
        n_in = self.n_in
        n_out = self.n_out

        with Pool(self.num_workers) as pool:
            # Run in parallel
            results = list(tqdm(pool.imap(self.compute_column, range(n_in)), total=n_in))

        # Collect results
        row_indices = []
        col_indices = []
        data = []

        for r, c, d in results:
            row_indices.extend(r)
            col_indices.extend(c)
            data.extend(d)

        # Create sparse matrix in COO format first, then convert
        A_sparse = coo_matrix((data, (row_indices, col_indices)), shape=(n_out, n_in)).tocsc()
        return A_sparse
        
    
    def get_range_space_basis(self, A_sparse, sigma_threshold_ratio = 0.001):
        # Compute a few smallest singular values
        p,q = A_sparse.shape
        kmax = int(min(p,q)-1)
        #t0 = time.time()
        umat, sing, vt = svds(A_sparse, k =kmax )  # smallest magnitude
        #t1 = time.time()
        #print(f'Took {t1-t0:2f} seconds to compute the SVD of A for kmax = {kmax}')

        # Null space basis vectors: columns of vt.T corresponding to near-zero singular values
        threshold = sigma_threshold_ratio * np.max(sing)
        range_space_basis = vt[sing >= threshold].T
        #print(f'Sigma 1 = {np.max(sing)}')
        return range_space_basis



    def make_null_projection_operator(self,range_basis):
        n = range_basis.shape[0]

        #def matvec(x):
        #    return null_basis @ (null_basis.T @ x)

        #return LinearOperator((n, n), matvec=matvec, dtype=null_basis.dtype)
        return  np.eye(n)- range_basis.dot(range_basis.T)


# Function to apply matrix transformation A to points
def apply_forwardmodel(A, points):
    return np.dot(points, A.T)

def projection_nullspace_operator(A):
    """Compute the matrix for projecting onto the null space of a matrix A, i.e. P_{N(A)}= (I - A^dagger A)
    Args: 
        - A: matrix 
    Returns:
        - project_ns: matrix projecting onto the null space of A.
    """
    A_dagger = np.linalg.pinv(A)
    project_ns= np.eye(A.shape[1]) - np.dot(A_dagger, A)
    return project_ns


def projection_nullspace(A, x):
    """
        Compute the projection of a point x onto the null space of A, i.e., P_{N(A)}(x).
        This is equivalent to (I - A^dagger A) x usin the function projection_nullspace for computing P_{N(A)} from A.
    """
    project_ns = projection_nullspace_operator(A)
    x_ns = np.dot(project_ns,x)
    
    return x_ns

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




def torch_csr_to_scipy(A: torch.Tensor) -> sp.csr_matrix:
    """"Convert matrix/linear forward model, as torch tensor, to scipy sparse matrix.
        Arguments:
            - A: torch tensor as matrix/forward model.
        Returns:
            - Scipy sparse matrix.
    """
    assert A.layout == torch.sparse_csr, f"A must be sparse_csr {type(A.layout)}"
    m, n = A.shape
    indptr = A.crow_indices().detach().cpu().numpy()   # length m+1
    indices = A.col_indices().detach().cpu().numpy()   # length nnz
    data = A.values().detach().cpu().numpy()           # length nnz

    # (optional) SciPy prefers int32 for indices
    if indices.dtype != np.int32: indices = indices.astype(np.int32, copy=False)
    if indptr.dtype  != np.int32: indptr  = indptr.astype(np.int32,  copy=False)

    sp_matrix = sp.csr_matrix((data, indices, indptr), shape=(m, n))

    return sp_matrix 

def torch_coo_to_scipy(A: torch.Tensor, to='csr'):
    A = A.to_sparse_coo().coalesce()
    m, n = A.shape
    ij = A.indices().detach().cpu().numpy()            # shape [2, nnz]
    data = A.values().detach().cpu().numpy()
    row, col = ij[0], ij[1]
    M = sp.coo_matrix((data, (row, col)), shape=(m, n))
    return M.tocsr() if to == 'csr' else M

def torch_sparse_to_scipy_csr(A: torch.Tensor) -> sp.csr_matrix:
    """
    Convert a PyTorch tensor (CSR/COO/dense) to SciPy csr_matrix.
    - No densifying for sparse inputs.
    - Handles CUDA tensors by moving index/data arrays to CPU.
    """
    m, n = A.shape

    # Dense (strided) tensor
    if A.layout == torch.strided:
        return sp.csr_matrix(A.detach().cpu().numpy())

    # Try CSR fast-path
    try:
        indptr  = A.crow_indices()   # only valid for CSR
        indices = A.col_indices()
        data    = A.values()
    except (AttributeError, RuntimeError):
        # Not CSR → go through COO safely
        Acoo = A.to_sparse_coo().coalesce()
        ij   = Acoo.indices().detach().cpu().numpy()  # [2, nnz]
        row, col = ij[0].astype(np.int32, copy=False), ij[1].astype(np.int32, copy=False)
        dat  = Acoo.values().detach().cpu().numpy()
        return sp.coo_matrix((dat, (row, col)), shape=(m, n)).tocsr()
    else:
        # CSR build (ensure int32 indices for SciPy)
        indptr_np  = indptr.detach().cpu().numpy().astype(np.int32, copy=False)
        indices_np = indices.detach().cpu().numpy().astype(np.int32, copy=False)
        data_np    = data.detach().cpu().numpy()
        return sp.csr_matrix((data_np, indices_np, indptr_np), shape=(m, n))
    