import numpy as np
from torch.utils.data import DataLoader
from playground.generator_functions import random_uni_points_in_ball
from src.accuracy_bounds.inverseproblems.feasible_sets import compute_feasible_set_linear_forwardmodel
from src.accuracy_bounds.inverseproblems.kersize_compute import worstcase_kernelsize, worstcase_kernelsize_sym, average_kernelsize, average_kernelsize_sym
from src.accuracy_bounds.inverseproblems.utils import apply_forwardmodel
from src.accuracy_bounds.inverseproblems.kersize_compute_dataloader import worstcase_kernelsize_appartenance, average_kernelsize_appartenance, target_distances_cuda_V2
from src.accuracy_bounds.inverseproblems.feasible_sets_dataloader import feasible_appartenance_additive_noise_dataloader_cuda, feasible_appartenance_additive_noise_cuda 
from src.accuracy_bounds.inverseproblems.utils import torch_sparse_to_scipy_csr, torch_csr_to_scipy



# Toy example 1
num_points = 3000
radius = 2
center = (0,0,0)
dim = 3
epsilon=1e-1
seed = 43

# Toy forward operator
A = np.diag([1, 1, 0])  # Transformation matrix


target_data = random_uni_points_in_ball(num_points=num_points, radius=radius+epsilon, center=center, dim=3)   
input_data = apply_forwardmodel(A, target_data)



#analytical wc kersize for ball around 0 with radius r+epsilon
wc_kernel_size_analytical = 2*radius+2*epsilon
#analytical av kersize for ball around 0 with radius r+epsilon sampled with uniform distribution
av_kernel_size_analytical = np.multiply((radius+epsilon),np.power(1/3,1/2))

# Distance measure
p_1=2
p_2=2
# Kernel Size order
p=2



# Set the range of k values
max_k = 1000
input_target_pairs = 3000

# Step 1: Compute feasible sets from the linear forward model and data:
feasible_sets_list = []
for y in input_data[:input_target_pairs]:
    feas_set_y = compute_feasible_set_linear_forwardmodel(A, y, target_data[:input_target_pairs], p_2, epsilon)
    feasible_sets_list.append(feas_set_y)


# Step 2: Compute worst-case kernel sizes:
worstcase_kersize = worstcase_kernelsize(feasible_sets_list, p_1, p)

error_worstcase = np.abs(worstcase_kersize-wc_kernel_size_analytical)

if error_worstcase << 0.3:
    print("worst case kernelsize np version passed")
else:
    print("worst case kernelsize np version not passed")

# Step 2.1: Compute worst-case kernel sizes with symmetry assumption:
worstcase_kersize_sym = worstcase_kernelsize_sym(A, input_data[:input_target_pairs], target_data, p_1, p_2, p, 2*epsilon)


error_worstcase_sym = np.abs(worstcase_kersize_sym-wc_kernel_size_analytical)

if error_worstcase_sym << 0.3:
    print("worst case kernelsize symmetric np version passed")
else:
    print("worst case kernelsize symmetric np version not passed")

# step 2.2 computation with cuda

max_k = 3000
batch_size = 100

input_data_k = input_data[0:max_k,:]
target_data_k = target_data[0:max_k,:]
input_loader1 = DataLoader(input_data_k, batch_size=batch_size, num_workers=batch_size, drop_last=False)
input_loader2 = DataLoader(input_data_k, batch_size=batch_size, num_workers=batch_size, drop_last=False)
target_loader1 = DataLoader(target_data_k, batch_size=batch_size, num_workers=batch_size, drop_last=False)
target_loader2 = DataLoader(target_data_k, batch_size=batch_size, num_workers=batch_size, drop_last=False)   

feasible_appartenance = feasible_appartenance_additive_noise_dataloader_cuda(input_loader1, input_loader2, p_Y=p_2, epsilon= epsilon)
feasible_appartenance = feasible_appartenance.to(dtype=torch.float32).to_sparse_coo()

distsXX = target_distances_cuda_V2(target_loader1, feasible_appartenance, p_X=p_1, batchsize= batch_size)
    
feasible_appartenance = torch_csr_to_scipy(feasible_appartenance.cpu().to_sparse_csr())
distsXX = torch_sparse_to_scipy_csr(distsXX)

wc_kersize_cuda = worstcase_kernelsize_appartenance(distsXX, feasible_appartenance)

error_worstcase_cuda = np.abs(wc_kersize_cuda -wc_kernel_size_analytical)

if error_worstcase_cuda << 0.3:

    print("worst case kernelsize cuda version passed")
else:
    print("worst case kernelsize cuda version not passed")


# Test average kernel size computations in the limit k to infty

average_kersize = average_kernelsize(feasible_sets_list, p_1, p)

error_average = np.abs(average_kersize - av_kernel_size_analytical)

if error_average << 0.3:
    print("average kernelsize np version passed")
else:
    print("average kernelsize np version not passed")

# Test average symmetric kernel size computations in the limit k to infty

average_kersize_sym = average_kernelsize_sym(A, input_data, target_data, p_1, p_2, p, epsilon)

error_average_sym = np.abs(average_kersize_sym - av_kernel_size_analytical)

if error_average_sym << 0.3:
    print("average kernelsize symmetric np version passed")
else:
    print("average kernelsize symmetric np version not passed")

# Test average kernel size cuda computations in the limit k to infty

average_kersize_cuda = average_kernelsize_appartenance(distsXX, feasible_appartenance, p_X=2)

error_average_cuda = np.abs(average_kersize_cuda - av_kernel_size_analytical)

if error_average_cuda << 0.3:
    print("average kernelsize cuda version passed")
else:
    print("average kernelsize cuda version not passed")
