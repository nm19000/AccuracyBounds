from accuracy_bounds.data.sentinel2 import SRDataset
from torch.utils.data import DataLoader
import torch
from scipy import sparse
from accuracy_bounds.inverseproblems.utils import torch_csr_to_scipy, torch_sparse_to_scipy_csr
from accuracy_bounds.inverseproblems.kersize_compute_dataloader import (target_distances_samplingYX_perbatch_cuda,
    kersize_samplingYX,
    avgLB_samplingYX,
    avgkersize_samplingYX)

from accuracy_bounds.utils.utils import build_S2_patched_dataset

import os
import json
import argparse



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="OpenSR Test Prediction Script")
    parser.add_argument('--save_folder',type = str ,help='Where do you want to save your feasibility appartenance and distsXX matrices to disk')
    parser.add_argument('--noise_level', type=int, default=4000, help='Noise level for the Kersize algorithm')
    parser.add_argument('--patch_size', type=int, default=12, help='Size of the patches to use for distance computation (default: 12)')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size to use for distance computation')
    #parser.add_argument('--DSHR', action = 'store_true', default=False, help='If activated, the Lr image as input will be DS(HR) where DS is a bilinear interpolation. Else, it will be the original LR image from the dataset (be careful, the original pair LR,HR doesnt match perfectly well, the DS is crosssensor)')

    args = parser.parse_args()

    


    #root_data_path = "/localhome/iaga_dv/Dokumente/sat_data/patched_crossproc_test"
    root_data_path = "/localhome/iaga_dv/Dokumente/sat_data/patched_crossproc"


    data_paths = {ps: os.path.join(root_data_path, f'PS{ps}') for ps in [8,12,16,20]}


    batch_size = args.batch_size
    patchsize_X = args.patch_size
    patchsize_Y = patchsize_X//4
    noise_level = args.noise_level
    results_fp = args.save_folder

    data_path = data_paths[patchsize_X]

    print(f'Parameters : \n - Noise level = {noise_level} \n  -Patch size X : {patchsize_X} \n - Batch size : {batch_size}')

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



        patchsizesX = [8,12,16,20]
        full_img_datapath = '/p/project1/hai_1013/sat_data/cross_processed'
        subdatasets = ['naip', 'spot', 'spain_crops', 'spain_urban']

        for PS_X in patchsizesX:
            print(f'Creating patched dataset for a patchsize of {PS_X}')
            build_S2_patched_dataset(patchsize_X = PS_X,img_dset_folder =  '/p/project1/hai_1013/sat_data/cross_processed', subdatasets  = subdatasets, out_dsfolder = f'/p/project1/hai_1013/sat_data/patched_crossproc/PS{PS_X}', labels = ('hr_data', 'lr_data'), border_X = 0, SR_factor = 4)
            print('Done')

