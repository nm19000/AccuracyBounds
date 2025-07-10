import os
import json
import torch
import rasterio
from tqdm import tqdm
from pdb import set_trace
from matplotlib import pyplot as plt
import time

import sys
sys.path.append(os.path.abspath('../opensr-test/opensr_test'))
sys.path.append(os.path.abspath('src/inverseproblems'))

from config import Config
from main import Metrics
from kersize_compute import (wc_kernelsize_nosym_batch_cuda, 
wc_kernelsize_sym_batch_cuda,
    wc_kernelsize_nosym_perbatch_cuda, 
    wc_kernelsize_nosym_crossbatch_cuda,
    av_kernelsize,
    wc_kernelsize,
    diams_feasibleset_inv_sym,
    diams_feasibleset_inv_sym,
    diams_feasibleset_inv,
    compute_feasible_set)


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

            

            
    pass

if __name__== '__main__':
    patchsize_X = 12
    patchsize_Y = patchsize_X//4
    border_X = 16
    border_Y = border_X//4
    batchsize = 2000


    dset_folder = '../sat_data/cross_processed'
    subdsets = ['spain_crops','naip' ,'spain_urban','spot']

    img_moments_path = os.path.join(dset_folder, 'reflectance_moments.json')
    with open(img_moments_path, 'r') as f:
        img_moments_dic = json.load(f)

    distY_jsonpath = os.path.join(dset_folder, 'distY_distr.json')
    # Load the distribution data from the JSON file
    with open(distY_jsonpath, 'r') as f:
        distY_distr = json.load(f)
    

    # Define the benchmark experiment
    config = Config()
    metrics = Metrics(config=config)

    img_moments_path = os.path.join(dset_folder, 'reflectance_moments.json')
    with open(img_moments_path, 'r') as f:
        img_moments_dic = json.load(f)

    distY_jsonpath = os.path.join(dset_folder, 'distY_distr.json')
    # Load the distribution data from the JSON file
    with open(distY_jsonpath, 'r') as f:
        distY_distr = json.load(f)

    batchsizes = [50,100, 300, 500, 1000, 1500, 2000, 2500]


    for subds in subdsets:    
        data_folder = os.path.join(dset_folder, subds)
        img_folders = os.listdir(data_folder)
        bar = img_folders
        img_moments = img_moments_dic[subds]
        mu0, sigma0 = torch.tensor(distY_distr[subds]['mean']), torch.tensor(distY_distr[subds]['sigma'])
        opt_alpha = mu0/img_moments[0]

        noise_level = mu0 + 2*sigma0

        for idxstr in tqdm(bar):
            img_folder = os.path.join(data_folder, idxstr)
            lr_path = f'{img_folder}/lr_res.tif'
            sr_path = f'{img_folder}/sr_res.tif'
            hr_path = f'{img_folder}/hr_res.tif'

            with rasterio.open(hr_path) as hr_src:
                hr_img = hr_src.read()
            hr_img = torch.from_numpy(hr_img) 
            with rasterio.open(lr_path) as lr_src:
                lr_img = lr_src.read()
            lr_img = torch.from_numpy(lr_img) 
            with rasterio.open(sr_path) as sr_src:
                sr_img = sr_src.read()
            sr_img = torch.from_numpy(sr_img)     

            # Setup the LR, SR, and HR images
            metrics.setup(lr=lr_img, sr=sr_img, hr=hr_img)
            metrics.sr_harm_setup()

            patched_lr = get_patches_from_S2(lr_img, patchsize=patchsize_Y, stride=2, border=border_Y)
            patched_hr = get_patches_from_S2(hr_img, patchsize_X,8, border_X)
            
        
        
            KS = wc_kernelsize_nosym_perbatch_cuda(0, patched_lr, patched_hr, p_X=2, p_Y=2, epsilon = noise_level*(patchsize_Y**2), batch_size= batchsize)/(patchsize_X**2)

            print(f'Kernelsize = {KS:2f}')
            

