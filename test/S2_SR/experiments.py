from accuracy_bounds.data.sentinel2 import SRDataset
from torch.utils.data import DataLoader
import torch
from accuracy_bounds.inverseproblems.utils import torch_csr_to_scipy, torch_sparse_to_scipy_csr
from accuracy_bounds.inverseproblems.kersize_compute_dataloader import (target_distances_samplingYX_perbatch_cuda,
    kersize_samplingYX,
    avgLB_samplingYX,
    avgkersize_samplingYX)

import os
import json


def get_patches_from_S2(img,patchsize,stride, border):
    h,w = img.shape[1:]
    n_patches_y = (h-border-border-patchsize)//stride
    n_patches_x = (w-border-patchsize)//stride


    all_patches = []
    for i in range(n_patches_y):
        imin = border + i*stride
        imax = imin + patchsize
        for j in range(n_patches_x):
            jmin = border + j*stride
            jmax = jmin + patchsize

            all_patches.append(img[:, imin:imax, jmin:jmax])
    
    all_patches = torch.stack(all_patches)
    return all_patches

if __name__ =='__main__':


    data_path_test = "/localhome/iaga_dv/Dokumente/sat_data/patched_crossproc_test"

    data_paths = {12:"/localhome/iaga_dv/Dokumente/sat_data/patched_crossproc/PS12_res"}

    batch_size = 200
    patchsize_X = 12
    patchsize_Y = patchsize_X//4

    data_path = data_paths[patchsize_X]

    noise_levels = {'naip': 2000, 'spain_crops':4100, 'spot': 5200, 'spain_urban':8800, 'global':3900} # from the 0.98 quantile of distY for non hall pixels



    distY_jsonpath = os.path.join('/localhome/iaga_dv/Dokumente/sat_data/cross_processed/naip', 'distY_distr.json')
    # Load the distribution data from the JSON file
    with open(distY_jsonpath, 'r') as f:
        distY_distr = json.load(f)

    noise_level = noise_levels['global']
    print(f'Noise level = {noise_level}')
    print()



    input_data = SRDataset(folder_path=data_path, suffixes=('lr_res'))
    target_data1 = SRDataset(folder_path=data_path, suffixes=('hr_res'))
    target_data2 = SRDataset(folder_path=data_path, suffixes=('hr_res'))
    forwarded_target_data = SRDataset(folder_path=data_path, suffixes=('lr_res'))

    print("Prepared Datasets")
    print("    input_data:           ", len(input_data))
    print("    target_data1:          ", len(target_data1))
    print("    target_data:2          ", len(target_data2))
    print("    forwarded_target_data:", len(forwarded_target_data))


    input_loader = DataLoader(input_data, batch_size=batch_size, num_workers=batch_size, drop_last=True, shuffle=False) # shuffle = False is important to keep the right order in the feasibility matrices
    target_loader1 = DataLoader(target_data1, batch_size=batch_size, num_workers=batch_size, drop_last=True, shuffle=False)
    target_loader2 = DataLoader(target_data2, batch_size=batch_size, num_workers=batch_size, drop_last=True, shuffle=False)
    forwarded_target_loader = DataLoader(forwarded_target_data, batch_size=batch_size, num_workers=batch_size, drop_last=True,shuffle=False)

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

    if False:
        from matplotlib import pyplot as plt
        from pdb import set_trace
        import numpy as np
        feasible_viz = feasible_appartenance.toarray()

        nb_feasible = np.sum(feasible_viz, axis=0)
        nonzer = nb_feasible>1.5
        print(f'For {nonzer.sum()}/{nb_feasible.shape[0]} y elements, there are more than 2 elements in F_y')
        plt.hist(nb_feasible, bins=20)
        plt.title(f'Sizes of the feasibles sets \n mean = {np.mean(nb_feasible)}\n std = {np.nanstd(nb_feasible)}')
        plt.show()

        plt.imshow(feasible_appartenance)
        plt.show()
        
    print("Compute WC Kernelsize")
    wc_kersize = kersize_samplingYX(distsXX, feasible_appartenance, p_X = 1)/(patchsize_X**2)


    print("Compute Avg. Kernelsize")
    avg_kersize = avgkersize_samplingYX(distsXX, feasible_appartenance, p_X = 1)/(patchsize_X**2)

    print(f'WC Kernelsize = {wc_kersize}')
    print(f'Avg. Kernelsize = {avg_kersize}')