import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import svds
import torch
import scipy.sparse as sp
import numpy as np

def torch_csr_to_scipy(A: torch.Tensor) -> sp.csr_matrix:
    assert A.layout == torch.sparse_csr, f"A must be sparse_csr {type(A.layout)}"
    m, n = A.shape
    indptr = A.crow_indices().detach().cpu().numpy()   # length m+1
    indices = A.col_indices().detach().cpu().numpy()   # length nnz
    data = A.values().detach().cpu().numpy()           # length nnz

    # (optional) SciPy prefers int32 for indices
    if indices.dtype != np.int32: indices = indices.astype(np.int32, copy=False)
    if indptr.dtype  != np.int32: indptr  = indptr.astype(np.int32,  copy=False)

    return sp.csr_matrix((data, indices, indptr), shape=(m, n))

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