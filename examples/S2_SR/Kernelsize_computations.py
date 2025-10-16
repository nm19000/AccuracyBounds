import os
import json
import argparse
import time
import rasterio
from scipy import sparse

from S2_SR_data import SRDataset, SRDataset_lightload,S2_Dataloader
from torch.utils.data import DataLoader
from src.accuracy_bounds.inverseproblems.feasible_sets_dataloader import feasible_appartenance_additive_noise_dataloader_cuda,feasible_appartenance_additive_noise_cuda 
from src.accuracy_bounds.inverseproblems.kersize_compute_dataloader import target_distances_cuda_V2
from src.accuracy_bounds.inverseproblems.utils import torch_sparse_to_scipy_csr, torch_csr_to_scipy

    
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="OpenSR Test Prediction Script")
    parser.add_argument('--data_path',type = str ,help='The folder you store your data in')
    parser.add_argument('--save_folder',type = str ,help='Where do you want to save your feasibility appartenance and distsXX matrices to disk')
    parser.add_argument('--noise_level', type=int, default=4000, help='Noise level for the Kersize algorithm')
    parser.add_argument('--patch_size', type=int, default=16,help='Size of the patches to use for distance computation (default: 16')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size to use for distance computation')
    parser.add_argument('--light_load', action='store_true', default=True)
    #parser.add_argument('--DSHR', action = 'store_true', default=False, help='If activated, the Lr image as input will be DS(HR) where DS is a bilinear interpolation. Else, it will be the original LR image from the dataset (be careful, the original pair LR,HR doesnt match perfectly well, the DS is crosssensor)')

    args = parser.parse_args()

    


    #root_data_path = "/localhome/iaga_dv/Dokumente/sat_data/patched_crossproc_test"
    #root_data_path = "/localhome/iaga_dv/Dokumente/sat_data/patched_crossproc"


    #data_paths = {ps: os.path.join(root_data_path, f'PS{ps}') for ps in [8,12,16,20]}


    batch_size = args.batch_size
    patchsize_X = args.patch_size
    patchsize_Y = patchsize_X//4
    noise_level = args.noise_level
    results_fp = args.save_folder
    data_path = args.data_path
    light_load = args.light_load

    patched_shapes12 = {'naip':(40,40), 'spain_crops': (42,42), 'spain_urban': (42,42), 'spot': (42,42)} 
    patched_shapes16 = {'naip':(30,30), 'spain_crops': (32,32), 'spain_urban': (32,32), 'spot': (32,32)} 
    patched_shapes20 = {'naip':(24,24), 'spain_crops': (25,25), 'spain_urban': (25,25), 'spot': (25,25)} 

    patched_shapes_all = {12: patched_shapes12, 16:patched_shapes16, 20:patched_shapes20}


    print(f'Parameters : \n - Noise level = {noise_level} \n  -Patch size X : {patchsize_X} \n - Batch size : {batch_size}')
    n_bands = 3
    print()
    print(f'Save folder = {results_fp}')
    print(f'Data path : {data_path}')

    if not light_load:
        input_data = SRDataset(folder_path=data_path, suffixes=('DSHR_res'))
        target_data1 = SRDataset(folder_path=data_path, suffixes=('hr_res'))
        target_data2 = SRDataset(folder_path=data_path, suffixes=('hr_res'))
        forwarded_target_data = SRDataset(folder_path=data_path, suffixes=('DSHR_res'))
        print("Prepared Datasets")
        print("    input_data:           ", len(input_data))
        print("    target_data1:          ", len(target_data1))
        print("    target_data:2          ", len(target_data2))
        print("    forwarded_target_data:", len(forwarded_target_data))
    else:
        input_data_light = SRDataset_lightload(data_path= data_path, patchsizes=(patchsize_Y,patchsize_X), suffixes=('DSHR_res'), patched_shapes=patched_shapes_all[patchsize_X])
        target_data_light = SRDataset_lightload(data_path= data_path, patchsizes=(patchsize_Y,patchsize_X), suffixes=('hr_res'), patched_shapes=patched_shapes_all[patchsize_X])
        print("    input_data_light:", len(input_data_light))
        print("    target_data_light:", len(target_data_light))

    import multiprocessing
    # Logical cores (includes hyperthreading)
    logical_cores = multiprocessing.cpu_count()
    print(f"Logical CPU cores: {logical_cores}")

    if not light_load:
        input_loader = DataLoader(input_data, batch_size=batch_size, num_workers=logical_cores//2, drop_last=False, shuffle=False) # shuffle = False is important to keep the right order in the feasibility matrices
        target_loader1 = DataLoader(target_data1, batch_size=batch_size, num_workers=logical_cores//2, drop_last=False, shuffle=False)
        target_loader2 = DataLoader(target_data2, batch_size=batch_size, num_workers=logical_cores//2, drop_last=False, shuffle=False)
        forwarded_target_loader = DataLoader(forwarded_target_data, batch_size=batch_size, num_workers=logical_cores//2, drop_last=False,shuffle=False)

        print("Prepared DataLoaders")
        print("    input_loader:           ", len(input_loader), "batches with size", batch_size)
        print("    target_loader1:          ", len(target_loader1), "batches with size", batch_size)
        print("    target_loader2:          ", len(target_loader2), "batches with size", batch_size)
        print("    forwarded_target_loader:", len(forwarded_target_loader), "batches with size", batch_size)
    else:
        input_loader_light = S2_Dataloader(input_data_light, suffix='DSHR_res', batch_size = batch_size, num_workers = logical_cores//2, drop_last = False, shuffle = False)
        forwarded_target_loader_light = S2_Dataloader(input_data_light, suffix='DSHR_res', batch_size = batch_size, num_workers = logical_cores//2, drop_last = False, shuffle = False)
        target_loader_light = S2_Dataloader(target_data_light, suffix='hr_res', batch_size = batch_size, num_workers = logical_cores//2, drop_last = False, shuffle = False)
        
        print("Prepared DataLoaders")    
        print("    input_loader_light:           ", len(input_loader_light), "batches with size", batch_size)
        print("    target_loader_light:          ", len(target_loader_light), "batches with size", batch_size)
        print("    forwarded_target_loader_light:          ", len(forwarded_target_loader_light), "batches with size", batch_size)



        
    if light_load:
        print("Compute Feasible Appartenance")
        feas_app = feasible_appartenance_additive_noise_dataloader_cuda(input_loader_light, forwarded_target_loader_light, p_Y=2, epsilon=(noise_level*n_bands*(patchsize_Y**2))**(1/2))
        #feas_app = feasible_appartenance_additive_noise_cuda(input_loader_light, forwarded_target_loader_light, p_Y = 2, epsilon=(noise_level*n_bands*(patchsize_Y**2))**(1/2), batchsize=1000000)
    
    else:
        print("Compute Feasible Appartenance")
        feas_app = feasible_appartenance_additive_noise_dataloader_cuda(input_loader, forwarded_target_loader, p_Y=2, epsilon=(noise_level*n_bands*(patchsize_Y**2))**(1/2))
    
    print("Convert to Scipy Sparse")
    feas_app_save = torch_csr_to_scipy(feas_app.cpu().to_sparse_csr())
    sparse.save_npz(os.path.join(results_fp, f'feas_app_PS{patchsize_X}_NL{noise_level}'), feas_app_save)

    distsXX =  target_distances_cuda_V2(target_loader_light, feas_app, p_X=1, batchsize=100000)

    
    distsXX = torch_sparse_to_scipy_csr(distsXX)
    sparse.save_npz(os.path.join(results_fp, f'distsXX_PS{patchsize_X}_NL{noise_level}'), distsXX)

