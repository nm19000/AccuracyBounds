import numpy as np
import torch 
from torch.utils.data import DataLoader
from playground.generator_functions import random_uni_points_in_ball
from accuracy_bounds.inverseproblems.feasible_sets import compute_feasible_set_linear_forwardmodel
from accuracy_bounds.inverseproblems.kersize_compute import worstcase_kernelsize, worstcase_kernelsize_sym, average_kernelsize, average_kernelsize_sym
from accuracy_bounds.inverseproblems.utils import apply_forwardmodel
from accuracy_bounds.inverseproblems.kersize_compute_dataloader import worstcase_kernelsize_appartenance, average_kernelsize_appartenance, target_distances_cuda_V2
from accuracy_bounds.inverseproblems.feasible_sets_dataloader import feasible_appartenance_additive_noise_dataloader_cuda, feasible_appartenance_additive_noise_cuda 
from accuracy_bounds.inverseproblems.utils import torch_sparse_to_scipy_csr, torch_csr_to_scipy


def test_worstcase_cuda():
    num_points = 3000
    radius = 2
    center = (0,0,0)
    epsilon=1e-1

    # Distance measure
    p_1=2
    p_2=2
    # Kernel Size order
    p=2

    # Toy forward operator
    A = np.diag([1, 1, 0])  # Transformation matrix

    target_data = random_uni_points_in_ball(num_points=num_points, radius=radius+epsilon, center=center, dim=3)   
    input_data = apply_forwardmodel(A, target_data)

    #analytical wc kersize for ball around 0 with radius r+epsilon
    wc_kernel_size_analytical = 2*radius+2*epsilon

    max_k = 3000
    batch_size = 100

    input_data_k = input_data[0:max_k,:]
    target_data_k = target_data[0:max_k,:]
    input_loader1 = DataLoader(input_data_k, batch_size=batch_size, num_workers=batch_size, drop_last=False)
    input_loader2 = DataLoader(input_data_k, batch_size=batch_size, num_workers=batch_size, drop_last=False)
    target_loader1 = DataLoader(target_data_k, batch_size=batch_size, num_workers=batch_size, drop_last=False)

    feasible_appartenance = feasible_appartenance_additive_noise_dataloader_cuda(input_loader1, input_loader2, p_Y=p_2, epsilon= epsilon)
    feasible_appartenance = feasible_appartenance.to(dtype=torch.float32).to_sparse_coo()

    distsXX = target_distances_cuda_V2(target_loader1, feasible_appartenance, p_X=p_1, batchsize= batch_size)
        
    feasible_appartenance = torch_csr_to_scipy(feasible_appartenance.cpu().to_sparse_csr())
    distsXX = torch_sparse_to_scipy_csr(distsXX)

    wc_kersize_cuda = worstcase_kernelsize_appartenance(distsXX, feasible_appartenance)

    error = np.abs(wc_kersize_cuda - wc_kernel_size_analytical)

    assert error < 0.3, f"Analytic Worstcase Kernel (Cuda) Error: {error}"

def test_average_kersize_cuda():
    num_points = 3000
    radius = 2
    center = (0,0,0)
    epsilon=1e-1

    # Distance measure
    p_1=2
    p_2=2

    # Toy forward operator
    A = np.diag([1, 1, 0])  # Transformation matrix

    target_data = random_uni_points_in_ball(num_points=num_points, radius=radius+epsilon, center=center, dim=3)   
    input_data = apply_forwardmodel(A, target_data)

    #analytical av kersize for ball around 0 with radius r+epsilon sampled with uniform distribution
    av_kernel_size_analytical = np.multiply((radius+epsilon),np.power(1/3,1/2))

    max_k = 3000
    batch_size = 100

    input_data_k = input_data[0:max_k,:]
    target_data_k = target_data[0:max_k,:]
    input_loader1 = DataLoader(input_data_k, batch_size=batch_size, num_workers=batch_size, drop_last=False)
    input_loader2 = DataLoader(input_data_k, batch_size=batch_size, num_workers=batch_size, drop_last=False)
    target_loader1 = DataLoader(target_data_k, batch_size=batch_size, num_workers=batch_size, drop_last=False)

    feasible_appartenance = feasible_appartenance_additive_noise_dataloader_cuda(input_loader1, input_loader2, p_Y=p_2, epsilon= epsilon)
    feasible_appartenance = feasible_appartenance.to(dtype=torch.float32).to_sparse_coo()

    distsXX = target_distances_cuda_V2(target_loader1, feasible_appartenance, p_X=p_1, batchsize= batch_size)
        
    feasible_appartenance = torch_csr_to_scipy(feasible_appartenance.cpu().to_sparse_csr())
    distsXX = torch_sparse_to_scipy_csr(distsXX)

    # Test average kernel size cuda computations in the limit k to infty
    average_kersize_cuda = average_kernelsize_appartenance(distsXX, feasible_appartenance, p_X=2)

    error = np.abs(average_kersize_cuda - av_kernel_size_analytical)

    assert error < 0.3, f"Analytic Average Kernel (Cuda) Error: {error}"

def test_feas_appartance_w_and_wo_dataloader():

    num_points = 3000
    radius = 2
    center = (0,0,0)
    epsilon=1e-1

    # Toy forward operator
    A = np.diag([1, 1, 0])  # Transformation matrix

    target_data = random_uni_points_in_ball(num_points=num_points, radius=radius+epsilon, center=center, dim=3)   
    input_data = apply_forwardmodel(A, target_data)

    ## Test if feasible appartenance matrix computation versions produce the same results

    batch_size = 100

    input_loader1 = DataLoader(input_data, batch_size=batch_size, num_workers=batch_size, drop_last=False)
    input_loader2 = DataLoader(input_data, batch_size=batch_size, num_workers=batch_size, drop_last=False)

    feas_app_1 = feasible_appartenance_additive_noise_dataloader_cuda(input_data=input_loader1, forwarded_target= input_loader2, p_Y=2, epsilon=epsilon)
    feas_app_2 = feasible_appartenance_additive_noise_cuda(input_loader1, input_loader2, p_Y=2, epsilon= epsilon, batchsize=50)

    def _to_dense(x):
        return x.to_dense() if torch.is_tensor(x) and x.layout != torch.strided else x

    a = _to_dense(feas_app_1)
    b = _to_dense(feas_app_2).to(device=a.device, dtype=a.dtype)

    error_tensor = (a - b).abs()
    error = float(error_tensor.mean().item())  # scalar Python float

    assert error < 0.3, f"Feasible Appearance Error: {error}"


test_feas_appartance_w_and_wo_dataloader()