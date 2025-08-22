from matplotlib import pyplot as plt
import scipy.sparse as sp
from scipy.sparse import load_npz
import numpy as np  
from accuracy_bounds.data.sentinel2 import SRDataset
from torch.utils.data import DataLoader
from accuracy_bounds.inverseproblems.utils import torch_csr_to_scipy, torch_sparse_to_scipy_csr
from accuracy_bounds.inverseproblems.kersize_compute_dataloader import (target_distances_samplingYX_perbatch_cuda,get_feasible_info,
    kersize_samplingYX,
    avgLB_samplingYX,
    avgkersize_samplingYX)

from accuracy_bounds.utils.utils import ImgComparator

from scipy import sparse
import os
from pdb import set_trace




# Running the algo on the test dataset
if False:
    test_data_path = '/localhome/iaga_dv/Dokumente/sat_data/patched_crossproc_test'
    batch_size = 999
    noise_level = 4000

    results_fp = '/localhome/iaga_dv/Dokumente/sat_data/patched_crossproc_test'

    input_data = SRDataset(folder_path=test_data_path, suffixes=('lr_res'))
    target_data1 = SRDataset(folder_path=test_data_path, suffixes=('hr_res'))
    target_data2 = SRDataset(folder_path=test_data_path, suffixes=('hr_res'))
    forwarded_target_data = SRDataset(folder_path=test_data_path, suffixes=('lr_res'))

    
    patchsize_X = target_data1.patchsizeX
    patchsize_Y = patchsize_X//4


    print("Prepared Datasets")
    print("    input_data:           ", len(input_data))
    print("    target_data1:          ", len(target_data1))
    print("    target_data:2          ", len(target_data2))
    print("    forwarded_target_data:", len(forwarded_target_data))

    import multiprocessing
    # Logical cores (includes hyperthreading)
    logical_cores = multiprocessing.cpu_count()
    print(f"Logical CPU cores: {logical_cores}")

    input_loader = DataLoader(input_data, batch_size=batch_size, num_workers=logical_cores//2, drop_last=True, shuffle=False) # shuffle = False is important to keep the right order in the feasibility matrices
    target_loader1 = DataLoader(target_data1, batch_size=batch_size, num_workers=logical_cores//2, drop_last=True, shuffle=False)
    target_loader2 = DataLoader(target_data2, batch_size=batch_size, num_workers=logical_cores//2, drop_last=True, shuffle=False)
    forwarded_target_loader = DataLoader(forwarded_target_data, batch_size=batch_size, num_workers=logical_cores//2, drop_last=True,shuffle=False)

    print("Prepared DataLoaders")
    print("    input_loader:           ", len(input_loader), "batches with size", batch_size)
    print("    target_loader1:          ", len(target_loader1), "batches with size", batch_size)
    print("    target_loader2:          ", len(target_loader2), "batches with size", batch_size)
    print("    forwarded_target_loader:", len(forwarded_target_loader), "batches with size", batch_size)

    print("Compute Feasible Appartenance")
    distsXX, feasible_appartenance = target_distances_samplingYX_perbatch_cuda(0, input_loader, target_loader1, target_loader2, forwarded_target_loader, p_X=1, p_Y=2, epsilon = (noise_level*(patchsize_Y**2))**(1/2))

    print("Convert to Scipy Sparse")
    feasible_appartenance = torch_csr_to_scipy(feasible_appartenance)
    distsXX = torch_sparse_to_scipy_csr(distsXX)

    sparse.save_npz(os.path.join(results_fp, f'feas_app_PS{patchsize_X}_NL{noise_level}'), feasible_appartenance)
    sparse.save_npz(os.path.join(results_fp, f'distsXX_PS{patchsize_X}_NL{noise_level}'), distsXX)

#Checking whether there are big enough feasible sets

if False:
    feas_app_path = '/localhome/iaga_dv/Dokumente/sat_data/patched_crossproc/results/feas_app_PS20_NL4000.npz'

    feas_app = load_npz(feas_app_path)
    n,m = feas_app.shape
    rows, cols = feas_app.nonzero()
    # Find diagonal elements where row == col
    diagonal_indices = rows == cols
    non_diagonal_count = len(rows) - diagonal_indices.sum()
    #print(f"Number of non-diagonal non-zero elements: {non_diagonal_count}")


data_path = '/localhome/iaga_dv/Dokumente/sat_data/patched_crossproc_DSHR4/PS12'

#data_path = '/localhome/iaga_dv/Dokumente/sat_data/patched_crossproc_test'

data_path = '/localhome/iaga_dv/Dokumente/sat_data/patched_crossproc_test'

input_data = SRDataset(folder_path=data_path, suffixes=('lr_res'), patched_shape= (39,39))
target_data = SRDataset(folder_path=data_path, suffixes=('hr_res'), patched_shape = (39,39))

all_data = SRDataset(folder_path=data_path, suffixes=('lr_res', 'hr_res'), patched_shape=(39,39))

hr_img =  all_data.get_full_img('naip', 0, 'hr_res')
lr_img = all_data.get_full_img('naip', 0, 'lr_res')
if False:
    for i in range(1):
        lr_img = all_data.get_full_img('naip', i, 'lr_res')

        plt.imshow(lr_img.permute(1,2,0)/3000)
        plt.show()

        hr_img =  all_data.get_full_img('naip', i, 'hr_res')

        plt.imshow(hr_img.permute(1,2,0)/3000)
        plt.show()

feas_app = sparse.csr_matrix((3000,3000))
feas_app_true = sparse.load_npz(os.path.join(data_path, f'feas_app_PS12_NL4000.npz'))

n,p = feas_app_true.shape
feas_app[:n,:p]+= feas_app_true

distsXX = sparse.load_npz(os.path.join(data_path, f'distsXX_PS12_NL4000.npz'))

feas_info = get_feasible_info(distsXX, feas_app)

if False:
    pos_test_mat = np.zeros((39,39)) 
    for i in range(1521):
        a,b = all_data.get_patch_position(i)
        pos_test_mat[a,b] = int(i)
    print(pos_test_mat)


all_data.get_Fy_fullimg_V2('naip', '0', feas_info,feas_app, 'hr_res', 'lr_res')
Fy_lim1, Fy_lim2, Fy_lim1_Y, Fy_lim2_Y, cards_Fy = all_data.get_Fy_lim_fullimg('naip', '0', feas_info, 'hr_res')
#Fy_lim1, Fy_lim2, cards_Fy = all_data.get_Fy_lim_fullimg_poissonblending('naip', '0', feas_info, 'hr_res')


fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# First subplot: low-res image
axs[0, 0].imshow(hr_img.permute(1, 2, 0) / 3000)
axs[0, 0].set_title('Hig-Resolution Image')
axs[0, 0].axis('off')

# Second subplot: high-res image
im = axs[0, 1].imshow(cards_Fy)
fig.colorbar(im, ax=axs[0, 1], fraction=0.046, pad=0.04)
axs[0, 1].axis('off')

# Third subplot: feasible appartenance matrix (as image)
axs[1, 0].imshow(Fy_lim1.permute(1, 2, 0) / 3000)
axs[1, 0].set_title('Fy lim 1')
axs[1, 0].axis('off')

# Fourth subplot: distsXX matrix (as image)
axs[1, 1].imshow(Fy_lim2.permute(1, 2, 0) / 3000)
axs[1, 1].set_title('Fy lim 2')
axs[1, 1].axis('off')

comp = ImgComparator(fig)
plt.tight_layout()
plt.show()

