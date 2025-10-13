import json
import os
import multiprocessing
import rasterio
import torch
import numpy as np
from scipy import sparse
from S2_SR_data import SRDataset_perimg
from tqdm import tqdm
from utils import bilinear_SR, bicubic_SR
import pandas as pd
import matplotlib.pyplot as plt
from accuracy_bounds.inverseproblems.feasible_sets_dataloader import get_feasible_info

from opensr_test.main import Metrics
from opensr_test.config import Config



def metrics_opensrtest(data_path,feas_appartenance_patches , feas_info_patches):
    config = Config()
    metrics = Metrics(config=config)
    
    logical_cores = multiprocessing.cpu_count()
    print(f"Logical CPU cores: {logical_cores}")
    dataset_fullimg = SRDataset_perimg(folder_path=data_path,feasible_appartenance_patches=feas_appartenance_patches, feasible_information_patches=feas_info_patches, suffixes=('lr_res','hr_res'), patch_suffixes=('lr_res', 'hr_res'), data_augmentation_type='standardsym', patched_shapes=patched_shapes, P_null_path=P_null_path)

    images_idx_order = []
    # Computing the Distances between images in the same feasible set for the Kersizes computations
    K = len(dataset_fullimg.image_ids)

    #distsLoss = {'diff': {1:np.zeros(K), 2: np.zeros(K)}, 'bilin': {1:np.zeros(K), 2: np.zeros(K)}, 'bicub': {1:np.zeros(K), 2: np.zeros(K)}}

    results = {'diffusion': [], 'bilinear':[], 'bicubic':[]}
    model_names = ['diffusion', 'bilinear', 'bicubic']

    # 1*i is diffusion, 2*i is bilinear, 3*i is bicubic

    # For each image initially in the dataset, we find a feasible set of 6 elements : hr, lim1, lim2 and they symetric versions
    for i in tqdm(range(K)):
        img_id = dataset_fullimg.image_ids[i]
        subds, nb_img = img_id
        print(f'Image id = {dataset_fullimg.image_ids[i]}')

        images_idx_order.append(img_id)

        # Get the predictions from the different models : 
        img_folder = f'{preds_folder_path}/{subds}/{nb_img}'
        lr_path = f'{img_folder}/DSHR_res.tif'
        hr_path = f'{img_folder}/hr_res.tif'
        with rasterio.open(lr_path) as lr_src:
            lr_img = lr_src.read()
        lr_img = torch.from_numpy(lr_img)

        with rasterio.open(hr_path) as hr_src:
            hr_img = hr_src.read()
        hr_img = torch.from_numpy(hr_img)


        # Diffusion SR : 
        srdiff_path = f'{img_folder}/sr_res_fromDSHR.tif'
        with rasterio.open(srdiff_path) as sr_src:
            srdiff_img = sr_src.read()
        srdiff_img = torch.from_numpy(srdiff_img)
        c,h,w = srdiff_img.shape
        srdiff_img = srdiff_img

        # Bilinear SR : 
        srbilin_img = bilinear_SR(lr_img, scale=4)
        c,h,w = srbilin_img.shape
        srbilin_img = srbilin_img

        # Bicubic SR : 
        srbicub_img = bicubic_SR(lr_img, scale=4)
        c,h,w = srbicub_img.shape
        srbicub_img = srbicub_img

        predictions = [srdiff_img, srbilin_img, srbicub_img]

        
        

        for model_idx in range(3):
            correctness = metrics.correctness(lr=lr_img, sr=predictions[model_idx], hr=hr_img)
            consistency = metrics.consistency(lr=lr_img, sr=predictions[model_idx])
            synthesis = metrics.synthesis(lr=lr_img, sr=predictions[model_idx], hr=hr_img)

            results[model_names[model_idx]].append({'correctness': correctness, 'consistency':consistency,'synthesis':synthesis })
        
    for model in results:
        all_refl = {model: x['consistency']['reflectance'] for x in results[model]}
        all_spectral = {model: x['consistency']['spectral'] for x in results[model]}
        all_spatial = {model: x['consistency']['spatial'] for x in results[model]}

        all_synthesis = {model: x['synthesis']['synthesis'] for x in results[model]}

        all_hall = {model: x['correctness']['ha_metric'] for x in results[model]}
        all_om = {model: x['correctness']['om_metric'] for x in results[model]}
        all_im = {model: x['correctness']['im_metric'] for x in results[model]}

        print(f'Model {model}')
        print(f'reflectance : {np.mean(all_refl[model])},  spectral : {np.mean(all_spectral[model])}, spatial : {np.mean(all_spatial[model])}')
        print(f'synthesis : {np.mean(all_synthesis[model])}')
        print(f'hall : {np.mean(all_hall[model])},  omission : {np.mean(all_om[model])}, im : {np.mean(all_im[model])}')

    pass
 

def compute_LB_dists(data_path,preds_folder_path,feas_appartenance_patches, feas_info_patches, patched_shapes, distances_outfolder, PS_X, P_null_path):
    logical_cores = multiprocessing.cpu_count()
    print(f"Logical CPU cores: {logical_cores}")
    dataset_fullimg = SRDataset_perimg(folder_path=data_path,feasible_appartenance_patches=feas_appartenance_patches, feasible_information_patches=feas_info_patches, suffixes=('lr_res','hr_res'), patch_suffixes=('lr_res', 'hr_res'), data_augmentation_type='standardsym', patched_shapes=patched_shapes, P_null_path=P_null_path)

    images_idx_order = []
    # Computing the Distances between images in the same feasible set for the Kersizes computations
    K = len(dataset_fullimg.image_ids)

    distsLoss = {'diff': {1:np.zeros(6*K), 2: np.zeros(6*K)}, 'bilin': {1:np.zeros(6*K), 2: np.zeros(6*K)}, 'bicub': {1:np.zeros(6*K), 2: np.zeros(6*K)}}
    dists_LB = {1:np.zeros((6*K, 6*K)), 2: np.zeros((6*K, 6*K))} # DistsLB i,j will correspond to the distance between the image i and the image j, if they are in the same feasible set
    # If they are not, the coefficient will be 0
    # For each image initially in the dataset, we find a feasible set of 6 elements : hr, lim1, lim2 and they symetric versions
    for i in tqdm(range(K)):
        feas_set = []
        img_id = dataset_fullimg.image_ids[i]
        subds, nb_img = img_id
        print(f'Image id = {dataset_fullimg.image_ids[i]}')

        # Get the predictions from the different models : 
        img_folder = f'{preds_folder_path}/{subds}/{nb_img}'
        lr_path = f'{img_folder}/DSHR_res.tif'
        with rasterio.open(lr_path) as lr_src:
            lr_img = lr_src.read()
        lr_img = torch.from_numpy(lr_img)


        # Diffusion SR : 
        srdiff_path = f'{img_folder}/sr_res_fromDSHR.tif'
        with rasterio.open(srdiff_path) as sr_src:
            srdiff_img = sr_src.read()
        srdiff_img = torch.from_numpy(srdiff_img)
        c,h,w = srdiff_img.shape
        srdiff_img = srdiff_img[:, :h-h%PS_X, :w-w%PS_X]

        # Bilinear SR : 
        srbilin_img = bilinear_SR(lr_img, scale=4)
        c,h,w = srbilin_img.shape
        srbilin_img = srbilin_img[:, :h-h%PS_X, :w-w%PS_X]

        # Bicubic SR : 
        srbicub_img = bicubic_SR(lr_img, scale=4)
        c,h,w = srbicub_img.shape
        srbicub_img = srbicub_img[:, :h-h%PS_X, :w-w%PS_X]

        if False:
            #plot lr, SR diff, SR bilin, SR bicub
            fig, axes = plt.subplots(2,2)

            axes[0,0].imshow(lr_img.permute(1,2,0)/3000)
            axes[0,0].set_title('LR img')

            axes[0,1].imshow(srdiff_img.permute(1,2,0)/3000)
            axes[0,1].set_title(f'SR diff img \n Shape = {srdiff_img.shape}')

            axes[1,0].imshow(srbicub_img.permute(1,2,0)/3000)
            axes[1,0].set_title(f'SR Bicubic img  \n Shape = {srbicub_img.shape}')

            axes[1,1].imshow(srbilin_img.permute(1,2,0)/3000)
            axes[1,1].set_title(f'SR Bilinear img \n Shape = {srbilin_img.shape}')

            plt.show()


        for sub_i in range(6):
            idx = 6*i+sub_i
            try:
                res_dict = dataset_fullimg.__getitem__(idx)
            except Exception as e:
                titles = {0: 'hr_img', 1:'lim1', 2:'lim2', 3: 'hr_sym', 4: 'lim1_sym', 5: 'lim2_sym'}
                print(f'Didnt manage to retrieve the image {dataset_fullimg.image_ids[i]}_{titles[sub_i]}')
                print(e)
                res_dict = dataset_fullimg.__getitem__(6*i)
                
            feas_set.append(res_dict['hr_res'])
            images_idx_order.append(res_dict['name'])

        feas_set = torch.stack(feas_set, dim=0)

        for p_norm in [1,2]:
            for sub_i in range(6):
                idx_i = 6*i + sub_i

                c,h,w = feas_set[sub_i, :,border_pnull:-border_pnull, border_pnull:-border_pnull].shape
                N = c*h*w - 2*border_pnull*(h+w)+ 4*border_pnull**2
                dist_i_diff = (1/(N**(1/p_norm)))*torch.norm(feas_set[sub_i, :,border_pnull:-border_pnull, border_pnull:-border_pnull]-srdiff_img[:, border_pnull:-border_pnull, border_pnull:-border_pnull], p = p_norm )
                dist_i_bilin = (1/(N**(1/p_norm)))*torch.norm(feas_set[sub_i, :,border_pnull:-border_pnull, border_pnull:-border_pnull]-srbilin_img[:, border_pnull:-border_pnull, border_pnull:-border_pnull], p = p_norm )
                dist_i_bicub = (1/(N**(1/p_norm)))*torch.norm(feas_set[sub_i, :,border_pnull:-border_pnull, border_pnull:-border_pnull]-srbicub_img[:, border_pnull:-border_pnull, border_pnull:-border_pnull], p = p_norm )

                distsLoss['diff'][p_norm][idx_i] = dist_i_diff
                distsLoss['bilin'][p_norm][idx_i] = dist_i_bilin
                distsLoss['bicub'][p_norm][idx_i] = dist_i_bicub

                dists_lossdiff_outpath = f'{distances_outfolder}/dists_loss_diff_{PS_X}_{p_norm}norm.npy'
                dists_lossbilin_outpath = f'{distances_outfolder}/dists_loss_bilin_{PS_X}_{p_norm}norm.npy'
                dists_lossbicub_outpath = f'{distances_outfolder}/dists_loss_bicub_{PS_X}_{p_norm}norm.npy'

                np.save(dists_lossdiff_outpath, distsLoss['diff'][p_norm])
                np.save(dists_lossbilin_outpath, distsLoss['bilin'][p_norm])
                np.save(dists_lossbicub_outpath, distsLoss['bicub'][p_norm])

                for sub_j in range(6):
                    c,h,w = feas_set[sub_i, :,border_pnull:-border_pnull, border_pnull:-border_pnull].shape
                    N = c*h*w
                    dist_ij = (1/(N**(1/p_norm)))*torch.norm(feas_set[sub_i, :,border_pnull:-border_pnull, border_pnull:-border_pnull]-feas_set[sub_j, :,border_pnull:-border_pnull, border_pnull:-border_pnull], p = p_norm )
                    idx_i = 6*i + sub_i
                    idx_j = 6*i + sub_j
                    dists_LB[p_norm][idx_i, idx_j] = dist_ij

                    # Calculate the distance between GT and preds
            
            dists_LB__outpath = f'{distances_outfolder}/dists_LB_{PS_X}_{p_norm}norm.npy'
            np.save(dists_LB__outpath, dists_LB[p_norm])
            
            idx_order_path = f'{distances_outfolder}/idx_order_{PS_X}_{p_norm}norm.json'
            with open(idx_order_path, "w") as f:
                    json.dump(images_idx_order, f)
    pass


def get_LB_loss_points(dists_LB, dists_loss, p,enrichment_type, KS_type, idx_order, classification):
    a,b = dists_LB.shape
    n_img = a//6

    LBs = []
    losses = []
    img_idxs = []
    classes = []
    for i in range(n_img):
        if enrichment_type == 'sym':
            LB_matrix = dists_LB[6*i:6*(i+1), 6*i:6*(i+1)]
            loss_arr = dists_loss[6*i:6*(i+1)]
        elif enrichment_type =='lim12':
            LB_matrix = dists_LB[6*i:6*i+3,6*i:6*i+3]
            loss_arr = dists_loss[6*i:6*i+3]
        elif enrichment_type == 'None':
            LB_matrix = dists_LB[6*i:6*i+1, 6*i:6*i+1]
            loss_arr = dists_loss[6*i:6*i+1]
        else:
            raise Exception('Please select a valid data enrichment type')
        
        if KS_type == 'default':
            LB = 0.5*np.mean(LB_matrix**p)**(1/p)
            loss = np.mean(loss_arr**p)**(1/p)
        elif KS_type == 'sym':
            if enrichment_type != 'sym':
                raise Exception('Symetric kernelsize only compatible with symetric enrichment')
            
            v_norms = np.array([LB_matrix[j,j+3] for j in range(3)])/2
            LB = np.mean(v_norms**p)**(1/p)
            loss = np.mean(loss_arr**p)**(1/p)
        
        if LB > 0.001:
            LBs.append(LB)
            losses.append(loss)
            img_id = idx_order[6*i].split(',')
            img_id = img_id[0][2:-1], img_id[1][2:-1]
            img_idxs.append(str())
            classes.append(classification[str(img_id)])

    ovr_loss = np.mean( np.array(losses)**p )**(1/p)
    loss_restricted_array = np.array([dists_loss[6*j] for j in range(n_img) ])
    restr_loss = np.mean(loss_restricted_array**p)**(1/p)
    df = pd.DataFrame({'LB': LBs,'Loss': losses, 'img_idxs': img_idxs, 'class':classes})
    #df = pd.DataFrame({'LB': LBs,'Loss': losses, 'label':classes})
    ovr_LB = np.mean(np.array(LBs)**p)**(1/p) # Half of the kernel size
    return df, ovr_loss, restr_loss, ovr_LB



if __name__ == '__main__':
    Test = False # Whether we use a small test dataset or the full one
    DSHR = True
    light_loading = False
    PS_X = 16
    PS_Y = PS_X//4
    p_norm = 2
    SR_factor = 4
    #noise_level_KS = 1333 # has to be 4000 or 1333
    noise_level_KS = 4000 # has to be 4000 or 1333 (only noise levels for which the Kersize has been calculated)
    preload_feas_info = True
    border_pnull = 2
    pred_type = 'bicub'
    p = 2

    patched_shapes12 = {'naip':(40,40), 'spain_crops': (42,42), 'spain_urban': (42,42), 'spot': (42,42)} 
    patched_shapes16 = {'naip':(30,30), 'spain_crops': (32,32), 'spain_urban': (32,32), 'spot': (32,32)} 
    patched_shapes20 = {'naip':(24,24), 'spain_crops': (25,25), 'spain_urban': (25,25), 'spot': (25,25)} 

    patched_shapes_all = {12: patched_shapes12, 16:patched_shapes16, 20:patched_shapes20}

    root_folder = '/localhome/iaga_dv/Dokumente'

    #Checking whether there are big enough feasible sets

  

    for PS_X in [16]:
        patched_shapes = patched_shapes_all[PS_X]
        print(f'Computations for Patch size = {PS_X}')
        print(f'Model : {pred_type}')
        print(f'Light loading : {light_loading}')
        if DSHR:
            DSHR_suffix = f'DSHR{SR_factor}'
        else:
            DSHR_suffix = ''

 
        data_path = f'{root_folder}/sat_data/patched_crossproc_{DSHR_suffix}/PS{PS_X}'
        feas_app_lightl_path = f'{root_folder}/sat_data/cross_processed/results/NL{noise_level_KS}/feas_app_PS{PS_X}_NL{noise_level_KS}.npz'
        distsXX_ligntl_path = f'{root_folder}/sat_data/cross_processed/results/NL{noise_level_KS}/distsXX_PS{PS_X}_NL{noise_level_KS}.npz'
        json_lightl_path = f'{root_folder}/sat_data/cross_processed/results/NL{noise_level_KS}/feas_info_PS{PS_X}_NL{noise_level_KS}.json'

        feas_app_path = f'{root_folder}/sat_data/patched_crossproc_{DSHR_suffix}/results/NL{noise_level_KS}/feas_app_PS{PS_X}_NL{noise_level_KS}.npz'
        distsXX_path = f'{root_folder}/sat_data/patched_crossproc_{DSHR_suffix}/results/NL{noise_level_KS}/distsXX_PS{PS_X}_NL{noise_level_KS}.npz'
        json_path = f'{root_folder}/sat_data/patched_crossproc_{DSHR_suffix}/results/NL{noise_level_KS}/feas_info_PS{PS_X}_NL{noise_level_KS}.json'
        
        distances_outfolder = f'{root_folder}/sat_data/patched_crossproc_{DSHR_suffix}/results/NL{noise_level_KS}'
        preds_folder_path = f'{root_folder}/sat_data/cross_processed'
        P_null_path = f'{root_folder}/Operators/P_null_32.npy'
        dists_LB_path = f'{distances_outfolder}/dists_LB_{PS_X}_{p_norm}norm.npy'
        dists_loss_path = f'{distances_outfolder}/dists_loss_{pred_type}_{PS_X}_{p_norm}norm.npy'
        idx_order_path = f'{distances_outfolder}/idx_order_{PS_X}_{p_norm}norm.json'
        classif_path = f'{preds_folder_path}/classif_dataset.json'

        # Get the image ids in order to iterate afterwards
        subdatasets = ['naip', 'spain_crops', 'spain_urban', 'spot']
        image_ids = []
        for subds in subdatasets:
            images_id_subds = os.listdir(os.path.join(preds_folder_path, subds))
            images_id_subds = [(subds, x) for x in images_id_subds]
            for x in images_id_subds:
                image_ids.append(x)
        #image_folderpaths = [os.path.join(preds_folder_path, x[0], x[1]) for x in image_ids]


        if not light_loading:
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
            feas_app = sparse.load_npz(feas_app_lightl_path)
            distsXX = sparse.load_npz(distsXX_ligntl_path)

            if not preload_feas_info:
                feas_info = get_feasible_info(distsXX, feas_app)
                with open(json_lightl_path, "w") as f:
                    json.dump(feas_info, f)
            else:
                with open(json_lightl_path, "r") as f:
                    feas_info = json.load(f)
        
        with open(classif_path, "r") as f:
                classification = json.load(f)
     


        if False: # Tests for the consistency of the results
            metrics_opensrtest(data_path,feas_appartenance_patches = feas_app, feas_info_patches= feas_info)        
  
   
        if False: # Get the values for the Avg Lb and loss after having made the computations
            dists_LB = np.load(dists_LB_path)
            dists_loss = np.load(dists_loss_path)
            
            for KS_type in ['sym', 'default']:
                for enrichment_type in ['lim12', 'sym']:
                    if not (KS_type == 'sym' and enrichment_type == 'lim12'):
                        print()
                        print()
                        print(f'KS type : {KS_type} ,  enrichment type : {enrichment_type}  p = {p}' )
                        print()
                        with open(idx_order_path, "r") as f:
                            idx_order = json.load(f)
                        
                        df_LB_losses, ovr_loss, restr_loss, ovr_LB = get_LB_loss_points(dists_LB=dists_LB, dists_loss=dists_loss, p = p, enrichment_type=enrichment_type, KS_type=KS_type, idx_order=idx_order, classification = classification)
                        print(df_LB_losses.to_string(index=False))
                        print(f'Loss term computed over the enriched dataset : \n {ovr_loss}')
                        print(f'Loss term computed over the restricted dataset : \n {restr_loss}')
                        print(f'Half Kernelsize computed over the Overall dataset : \n {ovr_LB}')

        if False:
            # Do the computations for the LB and loss
            compute_LB_dists(data_path,preds_folder_path, feas_app, feas_info, patched_shapes, distances_outfolder, PS_X, P_null_path)
  




    