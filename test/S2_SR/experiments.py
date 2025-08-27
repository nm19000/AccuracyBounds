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

from accuracy_bounds.utils.utils import ImgComparator, apply_upsampling, get_distance

from scipy import sparse
import os
from pdb import set_trace
import torch
from matplotlib.colors import TwoSlopeNorm
import json

if __name__ == '__main__':
    Test = False # Whether we use a small test dataset or the full one
    DSHR = True
    PS_X = 12
    SR_factor = 4
    noise_level_KS = 4000
    preload_feas_info = True
    patched_shapes = {'naip':(40,40), 'spain_crops': (42,42), 'spain_urban': (42,42), 'spot': (42,42)} 


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

    if not Test:
        if DSHR:
            DSHR_suffix = f'DSHR{SR_factor}'
        else:
            DSHR_suffix = ''

        data_path = f'/localhome/iaga_dv/Dokumente/sat_data/patched_crossproc_{DSHR_suffix}/PS{PS_X}'
        feas_app_path = f'/localhome/iaga_dv/Dokumente/sat_data/patched_crossproc_{DSHR_suffix}/results/feas_app_PS{PS_X}_NL{noise_level_KS}.npz'
        distsXX_path = f'/localhome/iaga_dv/Dokumente/sat_data/patched_crossproc_{DSHR_suffix}/results/distsXX_PS{PS_X}_NL{noise_level_KS}.npz'
        json_path = f'/localhome/iaga_dv/Dokumente/sat_data/patched_crossproc_{DSHR_suffix}/results/feas_info_PS{PS_X}_NL{noise_level_KS}.json'

        feas_app = sparse.load_npz(feas_app_path)
        distsXX = sparse.load_npz(distsXX_path)

        if not preload_feas_info:
            feas_info = get_feasible_info(distsXX, feas_app)
            with open(json_path, "w") as f:
                json.dump(feas_info, f)
        else:
            with open(json_path, "r") as f:
                feas_info = json.load(f)
    else:
        # A dataset with PS_X = 12
        data_path = '/localhome/iaga_dv/Dokumente/sat_data/patched_crossproc_test'
        json_path = f'/localhome/iaga_dv/Dokumente/sat_data/patched_crossproc_test/feas_info_PS12_NL4000.json'


        feas_app = sparse.csr_matrix((3000,3000))
        feas_app_true = sparse.load_npz(os.path.join(data_path, f'feas_app_PS12_NL4000.npz'))

        n,p = feas_app_true.shape
        feas_app[:n,:p]+= feas_app_true

        distsXX = sparse.load_npz(os.path.join(data_path, f'distsXX_PS12_NL4000.npz'))
        if not preload_feas_info:
            feas_info = get_feasible_info(distsXX, feas_app)
            with open(json_path, "w") as f:
                json.dump(feas_info, f)
        else:
            with open(json_path, "r") as f:
                feas_info = json.load(f)


    

    input_data = SRDataset(folder_path=data_path, suffixes=('lr_res'), patched_shapes= patched_shapes)
    target_data = SRDataset(folder_path=data_path, suffixes=('hr_res'), patched_shapes = patched_shapes)

    all_data = SRDataset(folder_path=data_path, suffixes=('lr_res', 'hr_res'), patched_shapes=patched_shapes)

    subdataset = 'naip'
    for idx_img in range(1,10):
        hr_img =  all_data.get_full_img(subdataset, str(idx_img), 'hr_res')
        lr_img = all_data.get_full_img(subdataset, str(idx_img), 'lr_res')

        # Does not give realistic images
        #Fy_lim1, Fy_lim2, Fy_lim1_Y, Fy_lim2_Y = all_data.get_Fy_fullimg_V3('naip', '0', feas_info,feas_app,sigma_blend=1, hr_suffix= 'hr_res', lr_suffix='lr_res' )

        # The only method that gives realistic images yet
        Fy_lim1, Fy_lim2, Fy_lim1_Y, Fy_lim2_Y = all_data.get_Fy_fullimg_V2(subdataset,str(idx_img) , feas_info,feas_app, 'hr_res', 'lr_res',lim_area_ratio=0.9, n_iter_max_ratio=1.0,sigma_blend= 1, margin_blend=1 )

        Fy_lim1_Y = apply_upsampling(Fy_lim1, scale = 4)
        Fy_lim2_Y = apply_upsampling(Fy_lim2, scale = 4)

        #Fy_lim1, Fy_lim2, Fy_lim1_Y, Fy_lim2_Y, cards_Fy = all_data.get_Fy_lim_fullimg('naip', '0', feas_info, 'hr_res')

        #Fy_lim1, Fy_lim2, cards_Fy = all_data.get_Fy_lim_fullimg_poissonblending('naip', '0', feas_info, 'hr_res')



        fig, axs = plt.subplots(2, 2, figsize=(20, 20))

        axs[0,0].imshow(hr_img.permute(1, 2, 0) / 3000)
        axs[0,0].set_title('Hig-Resolution Image')
        axs[0,0].axis('off')

        lim1_lr_dist = get_distance(Fy_lim1, Fy_lim2, method = 'l1', agg_method = 'patch', patch_size=12)

        axs[0,1].imshow(lim1_lr_dist)
        axs[0,1].set_title('Difference between lim1 and lim2 Image')
        axs[0,1].axis('off')

        axs[1,0].imshow(Fy_lim1.permute(1, 2, 0) / 3000)
        axs[1,0].set_title('Fy lim 1')
        axs[1,0].axis('off')

        axs[1,1].imshow(Fy_lim2.permute(1, 2, 0) / 3000)
        axs[1,1].set_title('Fy lim 2')
        axs[1,1].axis('off')


        comp = ImgComparator(fig)
        plt.tight_layout()
        plt.show()



        fig, axs = plt.subplots(2, 2, figsize=(20, 20))

        axs[0,0].imshow(lr_img.permute(1, 2, 0) / 3000)
        axs[0,0].set_title('Low-Resolution Image')
        axs[0,0].axis('off')

        lim1_lr_dist = get_distance(Fy_lim1_Y, lr_img, method = 'l2', agg_method = 'patch', patch_size=3)
        noise_level = 4000

        vcenter = noise_level
        vmin = lim1_lr_dist.min()
        vmax = max(noise_level*1.0001, 7000)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        axs[0,1].imshow(lim1_lr_dist, cmap = 'bwr', norm = norm)
        axs[0,1].set_title('Difference in the Y space lim img 1')
        axs[0,1].axis('off')

        lim2_lr_dist = get_distance(Fy_lim2_Y, lr_img, method = 'l2', agg_method = 'patch', patch_size=3)



        axs[1,0].imshow(lim2_lr_dist, cmap = 'bwr', norm = norm)
        axs[1,0].set_title('Difference in the Y space lim img 2')
        axs[1,0].axis('off')

        axs[1,1].imshow(Fy_lim2_Y.permute(1, 2, 0) / 3000)
        axs[1,1].set_title('Fy lim 2 Y space')
        axs[1,1].axis('off')


        comp = ImgComparator(fig)
        plt.tight_layout()
        plt.show()

