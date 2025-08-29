import os
import glob
import torch
from torch.utils.data import Dataset
import rasterio
from accuracy_bounds.utils.utils import build_S2_patched_dataset, build_S2_patched_dataset_DSHR, apply_square_op_full, apply_square_op_small
import argparse
from pdb import set_trace
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage import label
from scipy.ndimage import binary_dilation, binary_closing
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix


# Make a SRDataset per image
# Takes as input the information per patch to build full images + images lim and p_null as well
# The indexes are made on hr_res and lr_res has the index of the corresponding hr_res index //3 (or 2 or 4 depending on how we enrich the dataset)

class SRDataset_perimg(Dataset):
    def __init__(self, folder_path,feasible_information_patches, feasible_appartenance_patches, suffixes=('lr', 'hr'), patch_suffixes = ('lr', 'hr'),data_augmentation_type = 'None', patched_shapes = {'naip':(40,40), 'spain_crops': (42,42), 'spain_urban': (42,42), 'spot': (42,42)}, P_null_path = '/localhome/iaga_dv/Dokumente/Operators/P_null_32.npy' ) :
        """
        Args:
            folder_path (str): Path to folder containing image patches.
            suffixes (tuple): Suffixes used in filenames for LR, HR.
        """
        def get_img_id(patch_id):
            splitted_patch_id = patch_id.split('_')
            idx_img = splitted_patch_id[-2]  # index of the image in the subdataset
            if len(splitted_patch_id)==4:
                subds = f'{splitted_patch_id[0]}_{splitted_patch_id[1]}'
            elif len(splitted_patch_id)==3:
                subds = splitted_patch_id[0]

            return subds, idx_img
        
        self.folder_path = folder_path
        self.suffix_list = suffixes if not isinstance(suffixes, str) else [suffixes]
        self.suffix_list_patches = patch_suffixes if not isinstance(suffixes, str) else [suffixes]
        if 'lr' in self.suffix_list_patches[0]:
            self.lr_suffix = self.suffix_list_patches[0]
            self.hr_suffix = self.suffix_list_patches[1]
        elif 'hr' in self.suffix_list_patches[0]:
            self.lr_suffix = self.suffix_list_patches[1]
            self.hr_suffix = self.suffix_list_patches[0]
        else:
            raise Exception(f"Unvalid patch suffixes given. \n You gave {self.suffix_list_patches}")

        self.feas_info_patches = feasible_information_patches
        self.feas_app_patches = feasible_appartenance_patches

        # Gather all _lr.tif files and derive basename (e.g., 0001_0001)
        self.file_list = sorted(glob.glob(os.path.join(folder_path, f'*_{self.suffix_list_patches[0]}.tif')))
        self.patched_shape = patched_shapes

        self.patch_ids = [os.path.basename(f).replace(f'_{self.suffix_list_patches[0]}.tif', '') for f in self.file_list]

        self.image_ids = list(set([ get_img_id(p_id) for p_id in self.patch_ids]))
        imgs_idx_list = {}
        for img_id in self.image_ids:
            subds, n_img = img_id

            img_idx_list = [(i, self.patch_ids[i].split('_')[-1]) for i in range(len(self.patch_ids)) if (f'{subds}_{n_img}_' in self.patch_ids[i]) ]
            img_idx_list = sorted(
                img_idx_list,
                key=lambda x: int(x[1])
            )
            img_idx_list = [int(x[0]) for x in img_idx_list]
            imgs_idx_list[img_id] = img_idx_list

        self.imgs_idx_list = imgs_idx_list
        self.patched_shape = patched_shapes

        
        if data_augmentation_type not in ['None','standard', 'symetric', 'standardsym']:
            raise Exception(f"Incorrect data augmentation type. \n You gave {data_augmentation_type} \n Supported ones : 'None','standard', 'symetric', 'standardsym' ")
        self.data_augm_type = data_augmentation_type
        len_factors = {'None': 1, 'standard': 3, 'symetric': 2, 'standardsym':6}
        self.len_factor = len_factors[data_augmentation_type]

        # Building a custom feasible appartenance matrix, but this time, the points are full images (not patches). 
        # We assume that in the initially given dataset, there is no non trivial feasible set (in practice, this is almost always verified on full images)
        custom_feas_app = np.zeros((self.len_factor* len(self.image_ids), len(self.image_ids))) # rows = x, cols = y
        for y_idx in range(len(self.image_ids)):
            custom_feas_app[self.len_factor*y_idx: self.len_factor*(y_idx+1), y_idx] = 1
        self.feas_app_fullimg = csr_matrix(custom_feas_app)


        if len(self.suffix_list_patches)==2:
            self.file_list1 = sorted(glob.glob(os.path.join(folder_path, f'*_{self.suffix_list[1]}.tif')))

            with rasterio.open(self.file_list1[0]) as src:
                patch_1 = torch.from_numpy(src.read())
                patchsize_1 = patch_1.shape[-1]
            with rasterio.open(self.file_list[0]) as src:
                patch_0 = torch.from_numpy(src.read())
                patchsize_0 = patch_0.shape[-1]

            self.patchsizeX = max(patchsize_0, patchsize_1)
            self.patchsizeY = min(patchsize_0, patchsize_1)
            self.n_bands = patch_0.shape[0]
            self.SR_factor = self.patchsizeX//self.patchsizeY
        

        else:
            raise Exception("You need to give 2 suffixes for patches. The same as for images")
        
        # Parameters for lim images generation
        self.lim_area_ratio=0.9
        self.n_iter_max_ratio=1.0
        self.sigma_blend= 1

        # Parameters in the kernel projection
        self.border = 16
        self.P_null = np.load(P_null_path)

        

    
    def __len__(self):
        if len(self.suffix_list)==1:
            if 'lr' in self.suffix_list[0]:
                return len(self.image_ids)
            elif 'hr' in self.suffix_list[0]:
                return self.len_factor* len(self.image_ids)
                
        elif len(self.suffix_list)==2:
            return self.len_factor * len(self.image_ids)

        else:
            raise Exception(f"Invalid set of suffixes. You gave {self.suffix_list} ")
        
    def __getitem__(self, idx):
        idx_base = idx//self.len_factor # idx of the iage in the base dataset
        image_id = self.image_ids[idx_base]
        subdataset, idx_in_subds = image_id

        # Reconstituer LR image
        lr_image = self.get_full_img(subdataset, idx_in_subds, suffix=self.lr_suffix)

        # Separer en fonction des data_augmentation types
            # Separer en fonction des suffixes
        if self.data_augm_type == 'None':
            if len(self.suffix_list)==1:
                if 'lr' in self.suffix_list[0]:
                    return {'name' : image_id, 'img': lr_image}

                elif 'hr' in self.suffix_list[0]:
                    hr_image = self.get_full_img(subdataset, idx_in_subds, suffix=self.hr_suffix)
                    return {'name' : image_id, 'img': hr_image}
                    
            elif len(self.suffix_list)==2:
                res_dict = {'name': image_id}
                hr_image = self.get_full_img(subdataset, idx_in_subds, suffix=self.hr_suffix)

                res_dict[self.lr_suffix] = lr_image
                res_dict[self.hr_suffix] = hr_image
                return res_dict

        elif self.data_augm_type == 'standard':
            # We want to avoid calculating img_lim1,2 whenever it is not necessary. This is why the order of disjunction is weird

            if len(self.suffix_list)==1 and 'lr' in self.suffix_list[0]:
                raise Exception("It makes no sense to ask for data augmentation and only ask for lr images...")
            elif len(self.suffix_list)==2 and idx%self.len_factor ==0:
                res_dict = {'name': image_id}
                hr_image = self.get_full_img(subdataset, idx_in_subds, suffix=self.hr_suffix)
                res_dict[self.lr_suffix] = lr_image
                res_dict[self.hr_suffix] = hr_image
                return res_dict
            elif len(self.suffix_list)==1 and idx%self.len_factor ==0 and 'hr' in self.suffix_list[0]:
                res_dict = {'name': image_id}
                hr_image = self.get_full_img(subdataset, idx_in_subds, suffix=self.hr_suffix)
                res_dict['img'] = hr_image
                return res_dict

            Fy_lim1, Fy_lim2, _,_ = self.get_Fy_fullimg_V2(subdataset,str(idx_in_subds) , self.feas_info_patches ,self.feas_app_patches, self.hr_suffix,self.lr_suffix, lim_area_ratio=self.lim_area_ratio, n_iter_max_ratio=self.n_iter_max_ratio,sigma_blend= self.sigma_blend, margin_blend=1 )


            if len(self.suffix_list)==1 and 'hr' in self.suffix_list[0]:
                if idx%self.len_factor ==1:
                    res_dict = {'name': f'{image_id}_lim1'}
                    res_dict['img'] = Fy_lim1
                elif idx%self.len_factor== 2:
                    res_dict = {'name': f'{image_id}_lim2'}
                    res_dict['img'] = Fy_lim2
                else:
                    raise Exception("Oups something went wrong, and it is on our side")
            elif len(self.suffix_list)==2:
                if idx%self.len_factor ==1:
                    res_dict = {'name': f'{image_id}_lim1'}
                    res_dict[self.hr_suffix] = Fy_lim1
                    res_dict[self.lr_suffix] = lr_image

                elif idx%self.len_factor== 2:
                    res_dict = {'name': f'{image_id}_lim2'}
                    res_dict[self.hr_suffix] = Fy_lim1
                    res_dict[self.lr_suffix] = lr_image
                return res_dict

        elif self.data_augm_type == 'standardsym':
            # We want to avoid calculating img_lim1,2 whenever it is not necessary. This is why the order of disjunction is weird

            if len(self.suffix_list)==1 and 'lr' in self.suffix_list[0]:
                raise Exception("It makes no sense to ask for data augmentation and only ask for lr images...")
            elif len(self.suffix_list)==2 and idx%self.len_factor ==0:
                res_dict = {'name': image_id}
                hr_image = self.get_full_img(subdataset, idx_in_subds, suffix=self.hr_suffix)
                res_dict[self.lr_suffix] = lr_image
                res_dict[self.hr_suffix] = hr_image
                return res_dict
            
            elif len(self.suffix_list)==1 and idx%self.len_factor ==0 and 'hr' in self.suffix_list[0]:
                res_dict = {'name': image_id}
                hr_image = self.get_full_img(subdataset, idx_in_subds, suffix=self.hr_suffix)
                res_dict['img'] = hr_image
                return res_dict
            
            Fy_lim1, Fy_lim2, _,_ = self.get_Fy_fullimg_V2(subdataset,str(idx_in_subds) , self.feas_info_patches ,self.feas_app_patches, self.hr_suffix,self.lr_suffix, lim_area_ratio=self.lim_area_ratio, n_iter_max_ratio=self.n_iter_max_ratio,sigma_blend= self.sigma_blend, margin_blend=1 )


            # Get the kernel projection of hr
            hr_null = apply_square_op_full(self.P_null, hr_image, out_2D_shape_op=(128,128), border = self.border)
            lim1_null = apply_square_op_full(self.P_null, Fy_lim1, out_2D_shape_op=(128,128), border = self.border)
            lim2_null = apply_square_op_full(self.P_null, Fy_lim2, out_2D_shape_op=(128,128), border = self.border)

            # Get hr -2 hr_null
            hr_sym = hr_image-2*hr_null
            lim1_sym = Fy_lim1-2*lim1_null
            lim2_sym = Fy_lim2-2*lim2_null

            
            # TODO finish correcting

            if len(self.suffix_list)==1 and 'hr' in self.suffix_list[0]:
                if idx%self.len_factor ==1:
                    res_dict = {'name': f'{image_id}_lim1'}
                    res_dict['img'] = Fy_lim1
                    return res_dict
                elif idx%self.len_factor== 2:
                    res_dict = {'name': f'{image_id}_lim2'}
                    res_dict['img'] = Fy_lim2
                    return res_dict
                elif idx%self.len_factor== 3:
                    res_dict = {'name': f'{image_id}_hrsym'}
                    res_dict['img'] = hr_sym
                    return res_dict
                    
                elif idx%self.len_factor== 4:
                    res_dict = {'name': f'{image_id}_lim1sym'}
                    res_dict['img'] = lim1_sym
                    return res_dict
                    
                elif idx%self.len_factor== 5:
                    res_dict = {'name': f'{image_id}_lim2sym'}
                    res_dict['img'] = lim2_sym
                    return res_dict
                else:
                    raise Exception("Oups something went wrong, and it is on our side")
                
            elif len(self.suffix_list)==2:
                if idx%self.len_factor ==1:
                    res_dict = {'name': f'{image_id}_lim1'}
                    res_dict[self.hr_suffix] = Fy_lim1
                    res_dict[self.lr_suffix] = lr_image
                    return res_dict

                elif idx%self.len_factor== 2:
                    res_dict = {'name': f'{image_id}_lim2'}
                    res_dict[self.hr_suffix] = Fy_lim1
                    res_dict[self.lr_suffix] = lr_image
                    return res_dict
                elif idx%self.len_factor== 3:
                    res_dict = {'name': f'{image_id}_hrsym'}
                    res_dict[self.hr_suffix] = hr_sym
                    res_dict[self.lr_suffix] = lr_image
                    return res_dict
                elif idx%self.len_factor== 4:
                    res_dict = {'name': f'{image_id}_lim1sym'}
                    res_dict[self.hr_suffix] = lim1_sym
                    res_dict[self.lr_suffix] = lr_image
                    return res_dict
                elif idx%self.len_factor== 5:
                    res_dict = {'name': f'{image_id}_lim2sym'}
                    res_dict[self.hr_suffix] = lim2_sym
                    res_dict[self.lr_suffix] = lr_image
                    return res_dict
                else:
                    raise Exception("Oups something went wrong, and it is on our side")

        elif self.data_augm_type == 'symetric':
            idx_base = idx//self.len_factor # idx of the iage in the base dataset
            image_id = self.image_ids[idx_base]
            subdataset, idx_in_subds = image_id

            # Get the hr and lr images
            lr_image = self.get_full_img(subdataset, idx_in_subds, suffix=self.lr_suffix)
            hr_image = self.get_full_img(subdataset,idx_in_subds, suffix=self.hr_suffix )

            # Get the kernel projection of hr
            hr_null = apply_square_op_full(self.P_null, hr_image, out_2D_shape_op=(128,128), border = self.border)
            
            # Get hr -2 hr_null
            hr_sym = hr_image-2*hr_null

            # add the images accordingly
            if len(self.suffix_list)==1 and 'lr' in self.suffix_list[0]:
                raise Exception("It makes no sense to ask for data augmentation and only ask for lr images...")
            elif len(self.suffix_list)==1 and 'hr' in self.suffix_list[0]:
                if idx%self.len_factor == 0:
                    res_dict = {'name': image_id}
                    res_dict['img'] = hr_image
                    return res_dict
                
                elif idx%self.len_factor == 1:
                    res_dict = {'name': f'{image_id}_sym'}
                    res_dict['img'] = hr_sym
                    return res_dict
                else:
                    raise Exception("Oups something went wrong, and it is on our side")
                
                
            elif len(self.suffix_list)==2:
                if idx%self.len_factor == 0:
                    res_dict = {'name': image_id}
                    res_dict[self.lr_suffix] = lr_image
                    res_dict[self.hr_suffix] = hr_image
                    return res_dict
                
                elif idx%self.len_factor == 1:
                    res_dict = {'name': f'{image_id}_sym'}
                    res_dict[self.hr_suffix] = hr_sym
                    return res_dict
                else:
                    raise Exception("Oups something went wrong, and it is on our side")

        





        
    def get_full_img(self,subdataset, img_idx, suffix):
        
        
        img_idx_list = self.imgs_idx_list[(subdataset, str(img_idx))]
        return self.recompose_image(img_idx_list, suffix, patched_shape= self.patched_shape[subdataset])
    
    def recompose_image(self,idx_list, suffix, patched_shape):
    
        patches = self.get_2D_patchgrid(idx_list, suffix, patched_shape)

        n_patches_y, n_patches_x, c, ps, ps = patches.shape
        reconstructed = patches.permute(2, 0, 3, 1, 4)
        reconstructed = reconstructed.reshape(c, n_patches_y * ps, n_patches_x * ps)

        return reconstructed
    
    def get_2D_patchgrid(self, idx_list, suffix, patched_shape):
        assert suffix in self.suffix_list_patches


        patches_id = [self.patch_ids[idx] for idx in idx_list]
        paths = [os.path.join(self.folder_path, f'{patch_id}_{suffix}.tif') for patch_id in patches_id]
        patches = []
        for path in paths:
            with rasterio.open(path) as src:
                patches.append(torch.from_numpy(src.read()))

        patches = torch.stack(patches) 
        c = patches.shape[1]
        ps = patches.shape[2]

        n_patches_y , n_patches_x= patched_shape
        patches = patches.view(n_patches_y, n_patches_x, c, ps, ps)

        return patches
        
    def amplify_mask(self, small_mask, size,patched_shape):
        nr,nc = patched_shape
        if size == 'big':
            ps = self.patchsizeX
        elif size == 'small':
            ps = self.patchsizeY
        else:
            print('Please choose a size among big or small')
            raise ValueError
        big_mask = np.zeros((nr*ps, nc*ps))

        for i in range(nr):
            for j in range(nc):
                big_mask[i*ps: (i+1)*ps, j*ps: (j+1)*ps] = small_mask[i,j]
        return big_mask

    def get_img_id(self,idx_patch):
        patch_id = self.patch_ids[idx_patch]
        splitted_patch_id = patch_id.split('_')
        idx_img = splitted_patch_id[-2]  # index of the image in the subdataset
        if len(splitted_patch_id)==4:
            subds = f'{splitted_patch_id[0]}_{splitted_patch_id[1]}'
        elif len(splitted_patch_id)==3:
            subds = splitted_patch_id[0]

        return subds, idx_img
    
    def fill_patch_info(self, values_list, patched_shape):
        nr,nc = patched_shape
        patch_info = torch.zeros((nr,nc))
        for idx in range(len(values_list)):
            i,j = self.get_patch_position(idx, patched_shape=patched_shape)
            patch_info[int(i),int(j)] = values_list[int(idx)]
        return patch_info

    def get_search_shift(self,position_base, position_lim, patched_shape_base, patched_shape_lim):
        i_base, j_base = position_base
        i_lim, j_lim = position_lim
        nr_base,nc_base = patched_shape_base
        nr_lim,nc_lim = patched_shape_lim


        y_min , y_max= -min(i_base, i_lim), min(nr_base-i_base, nr_lim-i_lim)
        x_min, x_max = -min(j_base, j_lim), min(nc_base-j_base, nc_lim-j_lim)
        return y_min, y_max, x_min, x_max

    def get_patch_position(self,idx_in_img, patched_shape):
        nr,nc = patched_shape
        row = idx_in_img//nc
        col = idx_in_img%nc
        return row, col

    def get_patch_idx_img(self, position, patched_shape):
        i,j = position
        nr,nc = patched_shape
        return i*nc + j
    

    def shift_mask(self,mask, k, l, patched_shape_dst):
        shifted_res = np.zeros_like(mask)
        
        rows, cols = mask.shape

        nr_dst, nc_dst = patched_shape_dst
        
        # Calculate the source and destination slice ranges
        src_row_start = max(0, -k)
        src_row_end = rows - max(0, k)
        
        src_col_start = max(0, -l)
        src_col_end = cols - max(0, l)
        
        dst_row_start = max(0, k)
        dst_row_end = rows - max(0, -k)
        
        dst_col_start = max(0, l)
        dst_col_end = cols - max(0, -l)
        
        shifted_res[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = mask[src_row_start:src_row_end, src_col_start:src_col_end]
        if rows <= nr_dst: # destination bigger than source
            shifted = np.zeros(patched_shape_dst)
            shifted[:rows, :cols] = shifted_res
        else: # destination smaller than source
            shifted = shifted_res[:nr_dst, nc_dst]
        
        return shifted
    
    def get_same_feasible_area(self, position_base, position_lim, search_shift_lim, feas_app, base_img_id, lim_img_id, patched_shape_lim, patched_shape_base):
        '''
        Given the area to search in img_lim, for each patch in this img_lim, 
        we check whether it belongd in the feasible of the corresponding Fy of the base img
        '''
        i_lim, j_lim = position_lim
        i_base, j_base = position_base

        mask_same_feasible_lim = np.zeros(patched_shape_lim)
        replacement_idx = np.zeros(patched_shape_lim)
        sh_ymin, sh_ymax, sh_xmin, sh_xmax = search_shift_lim

        for i in range(sh_ymin, sh_ymax):
            for j in range(sh_xmin, sh_xmax):
                pos_1D_base = self.get_patch_idx_img((i+ i_base, j+j_base), patched_shape_base)
                idx_base = self.imgs_idx_list[base_img_id][pos_1D_base]

                pos1D_lim = self.get_patch_idx_img((i+i_lim, j+j_lim), patched_shape_lim)
                idx_lim = self.imgs_idx_list[lim_img_id][pos1D_lim]
            
                mask_same_feasible_lim[i+i_lim, j+j_lim] = feas_app[idx_lim, idx_base]
                if mask_same_feasible_lim[i+i_lim, j+j_lim] >0.5:
                    replacement_idx[i+i_lim, j+j_lim] = idx_lim


                if i==0 and j==0 and feas_app[idx_lim, idx_base] ==0 and True:
                    
                    print()
                    patch_id_base = self.patch_ids[idx_base]

                    img_base_id = self.get_img_id(int(idx_base))
                    subds_img_base = img_base_id[0]

                    position_1D_base = int(patch_id_base.split('_')[-1])
                    position_2D_base = self.get_patch_position(position_1D_base, self.patched_shape[subds_img_base])
                    print('Unexpected behaviour in the image reconstitution')
                    print(f'Looked Base position : {position_2D_base}')
                    print(f'Looked indexes : {idx_lim}, {idx_base}')


        labeled_array, num_features = label(mask_same_feasible_lim)
        component_label = labeled_array[i_lim, j_lim]
   

        if component_label == 0:
            return  np.zeros_like(mask_same_feasible_lim, dtype=bool), np.zeros_like(mask_same_feasible_lim)
        

        mask_same_feasible_lim = labeled_array == component_label
        replacement_idx = replacement_idx * mask_same_feasible_lim
        return mask_same_feasible_lim, replacement_idx

    def get_Fy_fullimg_idx_V2(self, subdataset, img_idx_insubds, feasible_info,feas_app, lim_area_ratio = 0.6, n_iter_max_ratio = 0.4):
        img_idx_list = self.imgs_idx_list[(subdataset, img_idx_insubds)]

        # For each index in the image, get the F_yidx info and put them into variables
        Fy_infos = [feasible_info[idx] for idx in img_idx_list]

        diams_Fy = torch.tensor([info[0] for info in Fy_infos])
        Fy_lim1_idx = torch.tensor([info[1][0] for info in Fy_infos])
        Fy_lim2_idx = torch.tensor([info[1][1] for info in Fy_infos])
        Fy_cards = [info[2] for info in Fy_infos]



        Fy_lim1_idx_small = self.fill_patch_info(Fy_lim1_idx, self.patched_shape[subdataset]) # indexes of the first patch corresponding to the diam of the local Fy, arranged in 2D
        Fy_lim2_idx_small = self.fill_patch_info(Fy_lim2_idx, self.patched_shape[subdataset]) # Same for the second patch corresponding to diam Fy

        diams_Fy_small = self.fill_patch_info(diams_Fy, self.patched_shape[subdataset])

        patched_shape = self.patched_shape[subdataset]
        # Sorting the diams_Fy keeping track of the positions in 2D
        flat_diams = diams_Fy_small.flatten()
        sorted_values, sorted_indices = torch.sort(flat_diams, descending=True)
        nc = patched_shape[1]
        rows = sorted_indices // nc
        cols = sorted_indices % nc
        indices_2d_diams = torch.stack((rows, cols), dim=1)
            


        replace_lim1_idx = np.zeros(patched_shape) # Stating which patch in the base image should be replaced by which one (in terms of indexes)
        replace_lim2_idx = np.zeros(patched_shape)

        replace_lim1_order = np.zeros(patched_shape)  # The order according to which the patches should be replaces
        replace_lim2_order = np.zeros(patched_shape)
        
        replaced_prop_lim1 = 0
        replaced_prop_lim2 = 0

        n_iter = 1  # number of iterations
        n_replace = 1 # number of times the patch replacement will be applied (smaller or equal to n_iter)
        
        nr,nc = patched_shape
        structuring_element = np.array([[0, 1, 0],
                                        [1, 1, 1],
                                        [0, 1, 0]], dtype=bool)

        while n_iter < nr*nc* n_iter_max_ratio and replaced_prop_lim1 < lim_area_ratio and replaced_prop_lim2 <lim_area_ratio : 
            maxdiam_position = indices_2d_diams[n_iter, :] #position of the patch verifying the max diam Fy, in the base image
            maxdiam_position = int(maxdiam_position[0]), int(maxdiam_position[1])
            i_base, j_base = maxdiam_position

            if False:
                maxdiam_1D_pos = self.get_patch_idx_img(maxdiam_position)
                print(f'Info Fy : {Fy_infos[maxdiam_1D_pos]}')


            idx_Fy_lim1 = Fy_lim1_idx_small[i_base, j_base] # index of the corresponding 1st patch (call it P1) associated to diam Fy 
            idx_Fy_lim2 = Fy_lim2_idx_small[i_base, j_base] # same for second one (call it P2)

            if False:
                print()
                print(f'idx_lim1 normallly : {idx_Fy_lim1}')
                print(f'idx_lim2 normallly : {idx_Fy_lim2}')

                print(f'Base position normally : {maxdiam_position}')
                base_1D_pos = self.get_patch_idx_img(maxdiam_position)
                base_idx = self.imgs_idx_list[(subdataset, img_idx)][base_1D_pos]
                print(f'Base idx normally : {base_idx}')
                print()

            if False:
                lim_patch1_id = self.patch_ids[int(idx_Fy_lim1)]
                img_lim1_id = self.get_img_id(int(idx_Fy_lim1))
                patch_lim1_nb = lim_patch1_id.split('_')[-1]
                
                position_img_lim1 = self.get_patch_position(int(patch_lim1_nb))
                i_lim1, j_lim1 = position_img_lim1

                pos1D_lim1 = self.get_patch_idx_img((i_lim1, j_lim1))

                idx_lim1 = self.imgs_idx_list[img_lim1_id][pos1D_lim1]

                

                print(f'idx_lim1 after f f^-1 :{idx_lim1}')
                print(f'img_lim1 : {img_lim1_id}')

                lim_patch2_id = self.patch_ids[int(idx_Fy_lim2)]
                img_lim2_id = self.get_img_id(int(idx_Fy_lim2))
                patch_lim2_nb = lim_patch2_id.split('_')[-1]
                position_img_lim2 = self.get_patch_position(int(patch_lim2_nb))
                i_lim2, j_lim2 = position_img_lim2

                pos1D_lim2 = self.get_patch_idx_img((i_lim2, j_lim2))

                idx_lim2 = self.imgs_idx_list[img_lim2_id][pos1D_lim2]

                print(f'idx_lim2 after f f^-1 :{idx_lim2}')
                print(f'img_lim2 : {img_lim2_id}')
                

            lim_patch1_id = self.patch_ids[int(idx_Fy_lim1)] # the corresponding patch ids
            lim_patch2_id = self.patch_ids[int(idx_Fy_lim2)]

            patched_shape_base = self.patched_shape[subdataset]

            img_lim1_id = self.get_img_id(int(idx_Fy_lim1)) # The id of the image where P1 belongs (subdataset, img idx)
            subds_img_lim1 = img_lim1_id[0]
            patched_shape_lim1 = self.patched_shape[subds_img_lim1]
            patch_lim1_nb = lim_patch1_id.split('_')[-1]  # Its place on the image, in the 1d array
            position_img_lim1 = self.get_patch_position(int(patch_lim1_nb), patched_shape= patched_shape_lim1 ) # its pplace on the image, but in 2D
            i_lim1, j_lim1 = position_img_lim1
            search_shift_lim1 = self.get_search_shift(maxdiam_position,position_img_lim1, patched_shape_base, patched_shape_lim1) # The area we search on around the position of p1 on its image
            search_area_lim1 = search_shift_lim1[0]+i_lim1,search_shift_lim1[1]+i_lim1, search_shift_lim1[2]+j_lim1, search_shift_lim1[3]+j_lim1 # This area + it position onthe image = area on the image
            same_feas_area_lim1, replacement_idx = self.get_same_feasible_area(maxdiam_position, position_img_lim1, search_shift_lim1, feas_app, (subdataset, img_idx), img_lim1_id,patched_shape_lim1, patched_shape_base )
            same_feas_area_lim1_shifted, replacement_idx_shifted_lim1 = self.shift_mask(same_feas_area_lim1, i_base-i_lim1, j_base-j_lim1, patched_shape_dst=patched_shape_base), self.shift_mask(replacement_idx, i_base-i_lim1, j_base-j_lim1, patched_shape_dst=patched_shape_base)

            img_lim2_id = self.get_img_id(int(idx_Fy_lim2))
            subds_img_lim2 = img_lim2_id[0]
            patched_shape_lim2 = self.patched_shape[subds_img_lim2]
            patch_lim2_nb = lim_patch2_id.split('_')[-1]
            position_img_lim2 = self.get_patch_position(int(patch_lim2_nb), patched_shape=patched_shape_lim2)
            i_lim2, j_lim2 = position_img_lim2
            search_shift_lim2 = self.get_search_shift(maxdiam_position,position_img_lim2, patched_shape_base, patched_shape_lim2)
            search_area_lim2 = search_shift_lim2[0]+i_lim2,search_shift_lim2[1]+i_lim2, search_shift_lim2[2]+j_lim2, search_shift_lim2[3]+j_lim2
            same_feas_area_lim2, replacement_idx = self.get_same_feasible_area(maxdiam_position, position_img_lim2, search_shift_lim2, feas_app, (subdataset, img_idx), img_lim2_id, patched_shape_lim2, patched_shape_base)
            same_feas_area_lim2_shifted, replacement_idx_shifted_lim2 = self.shift_mask(same_feas_area_lim2, i_base-i_lim2, j_base-j_lim2, patched_shape_dst=patched_shape_base), self.shift_mask(replacement_idx, i_base-i_lim2, j_base-j_lim2, patched_shape_dst=patched_shape_base)


            # Verify whether there is an overlap between the dilation of the iterated mask and the mask for this mask
            iterative_mask_lim1 = replace_lim1_idx> 0.5
            iterative_mask_lim2 = replace_lim2_idx> 0.5

            dilated_iterative_mask_lim1 = binary_dilation(iterative_mask_lim1, structure=structuring_element)
            dilated_iterative_mask_lim2 = binary_dilation(iterative_mask_lim2, structure=structuring_element)

            intersection_lim1 = np.logical_and(same_feas_area_lim1_shifted,dilated_iterative_mask_lim1 )
            intersection_lim2 = np.logical_and(same_feas_area_lim2_shifted,dilated_iterative_mask_lim2 )

            if not np.any(intersection_lim1) and not np.any(intersection_lim2):
                # Fill the iterated replacement patch indexes
                replace_lim1_idx = np.where(same_feas_area_lim1_shifted >0.5, replacement_idx_shifted_lim1, replace_lim1_idx)
                replace_lim2_idx = np.where(same_feas_area_lim2_shifted >0.5, replacement_idx_shifted_lim2, replace_lim2_idx)
                
                # Fill the order of filling in the iterated orders of fillings
                replace_lim1_order = np.where(same_feas_area_lim1_shifted >0.5, n_replace, replace_lim1_order)
                replace_lim2_order = np.where(same_feas_area_lim2_shifted >0.5, n_replace, replace_lim2_order)


                n_replace +=1



            n_iter +=1

            closed_mask_lim1 = binary_closing(replace_lim1_order>0.5, structure = structuring_element)
            closed_mask_lim2 = binary_closing(replace_lim2_order>0.5, structure = structuring_element)


            replaced_prop_lim1 = np.sum(closed_mask_lim1>0.5)/(patched_shape[0] * patched_shape[1])
            replaced_prop_lim2 = np.sum(closed_mask_lim2>0.5)/(patched_shape[0] * patched_shape[1])

            #print(f'n_iter = {n_iter-1}     n_replace = {n_replace-1} \n replaced_prop_lim1 = {np.round(100*replaced_prop_lim1)}%    replaced_prop_lim2 = {np.round(100*replaced_prop_lim2)}%')

            if False and n_iter %100 ==0:
           
                fig, axs = plt.subplots(1, 2, figsize=(10, 10))
                axs = axs.flatten()

             
                axs[0].imshow(replace_lim1_order>0.5, cmap='gray')
                axs[0].set_title('Replaced areas lim 1 ')
                axs[1].imshow(replace_lim2_order>0.5, cmap='gray')
                axs[1].set_title('Replaced areas lim 2')

                plt.tight_layout()
                plt.show()
                pass

             # For eqch limit image, 
             # 1. Get the area of search using position in the imglim 

             #Within the condition that the patch is not in the masked area
             # 2.Check for which patches, the patch if imglim belongs to the corresponding feasible set in the base image (shift to take in account)        mask_lim1 = torch.zeros(self.patched_shape)
             # 3. Take the connected component around the patch in question and paste it to the basee image
        
        return replace_lim1_idx,replace_lim2_idx, replace_lim1_order, replace_lim2_order

    def get_Fy_fullimg_V2(self, subdataset, img_idx, feasible_info,feas_app, hr_suffix = 'hr_res', lr_suffix = 'lr_res',  lim_area_ratio = 0.9, n_iter_max_ratio = 1.0, sigma_blend = 1, margin_blend = 1):

        replace_lim1_idx,replace_lim2_idx, replace_lim1_order, replace_lim2_order = self. get_Fy_fullimg_idx_V2(subdataset = subdataset, img_idx = img_idx, feasible_info = feasible_info, feas_app= feas_app, lim_area_ratio=lim_area_ratio, n_iter_max_ratio=n_iter_max_ratio)
        replace_lim1_idx_flat,replace_lim2_idx_flat, replace_lim1_order_flat, replace_lim2_order_flat = replace_lim1_idx.flatten(),replace_lim2_idx.flatten(), replace_lim1_order.flatten(), replace_lim2_order.flatten()
        # Get the indexlist of the original image
        baseimg_idx_list = np.array(self.imgs_idx_list[(subdataset, img_idx)])

        # For v1 : fill the image directly with the index list
        # For v2 : use patch blending to fill effectively the lim images
        blending = True
        if not blending:
            # Fill the indexes arrays by replacing
            img_lim1_idx_flat = np.where(replace_lim1_order_flat>0.5,replace_lim1_idx_flat,baseimg_idx_list ).astype(int)
            img_lim2_idx_flat = np.where(replace_lim2_order_flat>0.5,replace_lim2_idx_flat,baseimg_idx_list ).astype(int)

            Fy_lim1_img = self.recompose_image(img_lim1_idx_flat, suffix = hr_suffix, patched_shape= self.patched_shape[subdataset])
            Fy_lim2_img = self.recompose_image(img_lim2_idx_flat, suffix= hr_suffix, patched_shape= self.patched_shape[subdataset])

            Fy_lim1_imgY = self.recompose_image(img_lim1_idx_flat, suffix = lr_suffix, patched_shape= self.patched_shape[subdataset])
            Fy_lim2_imgY = self.recompose_image(img_lim2_idx_flat, suffix= lr_suffix, patched_shape= self.patched_shape[subdataset])
        else:
            base_imgX = self.recompose_image(baseimg_idx_list, suffix=hr_suffix, patched_shape= self.patched_shape[subdataset])
            img_lim1_idx_flat = np.where(replace_lim1_order_flat>0.5,replace_lim1_idx_flat,baseimg_idx_list ).astype(int)
            img_lim2_idx_flat = np.where(replace_lim2_order_flat>0.5,replace_lim2_idx_flat,baseimg_idx_list ).astype(int)

            Fy_lim1_imgY = self.recompose_image(img_lim1_idx_flat, suffix = lr_suffix, patched_shape= self.patched_shape[subdataset])
            Fy_lim2_imgY = self.recompose_image(img_lim2_idx_flat, suffix= lr_suffix, patched_shape= self.patched_shape[subdataset])


            # Isolate the indexes corresponding to that order in the 2D array
            res_mask_lim1 = replace_lim1_order >0.5
            res_bigmask_lim1 = self.amplify_mask(res_mask_lim1, size = 'big', patched_shape=self.patched_shape[subdataset]) # This is the mask telling where the patches will be sticked
            

            # Erode the blurred mask by 1 pixel
            kernel = np.ones((3, 3), np.uint8)
            eroded_mask = cv2.erode(res_bigmask_lim1, kernel, iterations=1)
            # Gaussian convolution of the mask 
            blurred_mask = torch.tensor(gaussian_filter(eroded_mask, sigma = sigma_blend))
            


            Fy_lim1_img_noblend = self.recompose_image(img_lim1_idx_flat, suffix = hr_suffix, patched_shape = self.patched_shape[subdataset])
            Fy_lim1_img = blurred_mask * Fy_lim1_img_noblend + (1-blurred_mask)*base_imgX


            # Isolate the indexes corresponding to that order in the 2D array
            res_mask_lim2 = replace_lim2_order >0.5
            res_bigmask_lim2 = self.amplify_mask(res_mask_lim2, size = 'big', patched_shape=self.patched_shape[subdataset]) # This is the mask telling where the patches will be sticked

            # Erode the blurred mask by 1 pixel
            eroded_mask = cv2.erode(res_bigmask_lim2, kernel, iterations=1)
            # Gaussian convolution of the mask
            blurred_mask = torch.tensor(gaussian_filter(eroded_mask, sigma = sigma_blend))

            Fy_lim2_img_noblend = self.recompose_image(img_lim2_idx_flat, suffix = hr_suffix, patched_shape= self.patched_shape[subdataset])
            Fy_lim2_img = blurred_mask * Fy_lim2_img_noblend + (1-blurred_mask)*base_imgX

        return Fy_lim1_img, Fy_lim2_img, Fy_lim1_imgY, Fy_lim2_imgY

class SRDataset(Dataset):
    def __init__(self, folder_path, suffixes=('lr', 'hr'), patched_shapes = {'naip':(40,40), 'spain_crops': (42,42), 'spain_urban': (42,42), 'spot': (42,42)} ) :
        """
        Args:
            folder_path (str): Path to folder containing image patches.
            suffixes (tuple): Suffixes used in filenames for LR, HR.
        """
        def get_img_id(patch_id):
            splitted_patch_id = patch_id.split('_')
            idx_img = splitted_patch_id[-2]  # index of the image in the subdataset
            if len(splitted_patch_id)==4:
                subds = f'{splitted_patch_id[0]}_{splitted_patch_id[1]}'
            elif len(splitted_patch_id)==3:
                subds = splitted_patch_id[0]

            return subds, idx_img
        
        self.folder_path = folder_path
        self.suffix_list = suffixes if not isinstance(suffixes, str) else [suffixes]

        # Gather all _lr.tif files and derive basename (e.g., 0001_0001)
        self.file_list = sorted(glob.glob(os.path.join(folder_path, f'*_{self.suffix_list[0]}.tif')))

        self.patched_shape = patched_shapes


        #self.file_list = sorted([f'{folder_path}/{path}' for path in os.listdir(folder_path) if self.suffix_list[0] in path ])
        self.patch_ids = [os.path.basename(f).replace(f'_{self.suffix_list[0]}.tif', '') for f in self.file_list]

        self.image_ids = list(set([ get_img_id(p_id) for p_id in self.patch_ids]))

        imgs_idx_list = {}

        for img_id in self.image_ids:
            subds, n_img = img_id

            img_idx_list = [(i, self.patch_ids[i].split('_')[-1]) for i in range(len(self.patch_ids)) if (f'{subds}_{n_img}_' in self.patch_ids[i]) ]
            img_idx_list = sorted(
                img_idx_list,
                key=lambda x: int(x[1])
            )
            
            img_idx_list = [int(x[0]) for x in img_idx_list]
            
            imgs_idx_list[img_id] = img_idx_list
                
        self.imgs_idx_list = imgs_idx_list

        self.patched_shape = patched_shapes
        
        if len(self.suffix_list)==2:
            self.file_list1 = sorted(glob.glob(os.path.join(folder_path, f'*_{self.suffix_list[1]}.tif')))

            with rasterio.open(self.file_list1[0]) as src:
                patch_1 = torch.from_numpy(src.read())
                patchsize_1 = patch_1.shape[-1]
            with rasterio.open(self.file_list[0]) as src:
                patch_0 = torch.from_numpy(src.read())
                patchsize_0 = patch_0.shape[-1]

            self.patchsizeX = max(patchsize_0, patchsize_1)
            self.patchsizeY = min(patchsize_0, patchsize_1)

            self.n_bands = patch_0.shape[0]

            self.SR_factor = self.patchsizeX//self.patchsizeY
        elif len(self.suffix_list)==1:
            if 'lr' in self.suffix_list[0]:
                with rasterio.open(self.file_list[0]) as src:
                    patch_0 = torch.from_numpy(src.read())
                    self.patchsizeY = patch_0.shape[-1]
                    self.n_bands = patch_0.shape[0]
            elif 'hr' in self.suffix_list[0]:
                with rasterio.open(self.file_list[0]) as src:
                    patch_0 = torch.from_numpy(src.read())
                    self.patchsizeX = patch_0.shape[-1]
                    self.n_bands = patch_0.shape[0]
            else:
                print('lr or hr not in the suffix names. Please add one of them.')
                print(f'Your suffixes are {suffixes}')
        else:
            print('Only 1 or 2 suffixes are alloxed')
            print(f'You gave suffixes = {suffixes}')
            
        

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        patch_id = self.patch_ids[idx]
        
        if len(self.suffix_list) == 1: 
            img_path = os.path.join(self.folder_path, f'{patch_id}_{self.suffix_list[0]}.tif')

            with rasterio.open(img_path) as src:
                img = torch.from_numpy(src.read())
            
            return {
                    'name': patch_id,
                    'img': img
                }
        else:
            result_dict = {"name": patch_id}

            for suffix in self.suffix_list: 

                img_path = os.path.join(self.folder_path, f'{patch_id}_{suffix}.tif')

                with rasterio.open(img_path) as src:
                    result_dict[suffix] = torch.from_numpy(src.read())

            return result_dict
    def get_img_id(self,idx):
        patch_id = self.patch_ids[idx]
        splitted_patch_id = patch_id.split('_')
        idx_img = splitted_patch_id[-2]  # index of the image in the subdataset
        if len(splitted_patch_id)==4:
            subds = f'{splitted_patch_id[0]}_{splitted_patch_id[1]}'
        elif len(splitted_patch_id)==3:
            subds = splitted_patch_id[0]

        return subds, idx_img
    
    def fill_patch_uniform_img(self, values_list, patchsize, patched_shape):
        '''
        Here, the inputs are the values associated to each patch of a single image, in the right order
        and we reconstruct an array with the image size containing these values, repecting the patch correspondence
        '''
        unif_patches = []
        for x in values_list:
            unif_patches.append(x* torch.ones((patchsize, patchsize)))

        unif_patches = torch.stack(unif_patches) 


        n_patches_y, n_patches_x = patched_shape


        reconstructed = unif_patches.view(n_patches_y, n_patches_x, patchsize, patchsize)
        reconstructed = reconstructed.permute(0, 2, 1, 3).contiguous()
        reconstructed = reconstructed.view(n_patches_y * patchsize, n_patches_x * patchsize)

        return reconstructed
    
    def fill_patch_info(self, values_list, patched_shape):
        nr,nc = patched_shape
        patch_info = torch.zeros((nr,nc))
        for idx in range(len(values_list)):
            i,j = self.get_patch_position(idx, patched_shape=patched_shape)
            patch_info[int(i),int(j)] = values_list[int(idx)]
        return patch_info

    def get_patch_position(self,idx_in_img, patched_shape):
        nr,nc = patched_shape
        row = idx_in_img//nc
        col = idx_in_img%nc
        return row, col

    def get_patch_idx_img(self, position, patched_shape):
        i,j = position
        nr,nc = patched_shape
        return i*nc + j
    
    def recompose_image(self,idx_list, suffix, patched_shape):
        

        patches = self.get_2D_patchgrid(idx_list, suffix, patched_shape)

        n_patches_y, n_patches_x, c, ps, ps = patches.shape
        reconstructed = patches.permute(2, 0, 3, 1, 4)
        reconstructed = reconstructed.reshape(c, n_patches_y * ps, n_patches_x * ps)

        return reconstructed
    
    def get_2D_patchgrid(self, idx_list, suffix, patched_shape):
        assert suffix in self.suffix_list


        patches_id = [self.patch_ids[idx] for idx in idx_list]
        paths = [os.path.join(self.folder_path, f'{patch_id}_{suffix}.tif') for patch_id in patches_id]
        patches = []
        for path in paths:
            with rasterio.open(path) as src:
                patches.append(torch.from_numpy(src.read()))

        patches = torch.stack(patches) 
        c = patches.shape[1]
        ps = patches.shape[2]


        n_patches_y , n_patches_x= patched_shape

        patches = patches.view(n_patches_y, n_patches_x, c, ps, ps)

        return patches

    def generate_mask(self, patchsize):
        mask = np.ones((patchsize, patchsize), dtype=np.uint8) * 255 
        return mask
    

    def seamless_blending(self,patch1, patch2, mask, center):
        """
        Apply Poisson blending using OpenCV's seamlessClone for RGB images.
        - patch1: The patch to blend (source).
        - patch2: The base image (destination).
        - mask: Binary mask indicating where blending happens.
        - center: Center point for the cloning (usually the center of patch2).
        """


        n_bands = patch1.shape[0]

        patch1_ = patch1.transpose(1,2,0)
        patch2_ = patch2.transpose(1,2,0)
        result_ = np.zeros_like(patch2_)

        result_ = cv2.seamlessClone(patch1_, patch2_, mask, center, cv2.NORMAL_CLONE)

        result = result_.transpose(2,0,1)
        return result

    def stitch_patches(self,patches, patchsize, grid_size, base_image, pixel_range = 3000):
        """
        Stitch the nrxnc patches together using Poisson blending for multiple bands images.
        """

        patches_ = patches*255/pixel_range
        patches_ = patches_.astype(np.uint8)

        base_image_ = base_image*255/pixel_range
        base_image_ = base_image_.astype(np.uint8)


        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                patch = patches_[i, j]
                
                # Define the region of base_image where this patch will be inserted
                x_offset = j * patchsize
                y_offset = i * patchsize
                patch_position = (x_offset + patchsize // 2, y_offset + patchsize // 2)
                
                # Generate a mask for the current patch (usually rectangular)
                mask = self.generate_mask(patchsize)
                
                # Apply Poisson blending (seamless cloning)
                base_image_ = self.seamless_blending(patch, base_image_, mask, patch_position)
        
        base_image_ = base_image_.astype(np.float32)
        return base_image_*pixel_range/255
    
    def get_Fy_lim_fullimg_poissonblending(self, subdataset, img_idx, feasible_info, hr_suffix = 'hr_res'):
        # First get the sorted indexes related to the image
        img_idx_list = [(i, self.patch_ids[i].split('_')[-1]) for i in range(len(self.patch_ids)) if (f'{subdataset}_{img_idx}_' in self.patch_ids[i]) ]
        
        img_idx_list = sorted(
            img_idx_list,
            key=lambda x: int(x[1])
        )
        img_idx_list = [int(x[0]) for x in img_idx_list]


        # Compose the images per patch : 2 images corresponfing to the 2 that achieve diam(F_y) and the card of the F_y

        # For each index in the image, get the F_yidx info and put them into variables
        Fy_infos = [feasible_info[idx] for idx in img_idx_list]
        Fy_lim1_idx = [info[1][0] for info in Fy_infos]
        Fy_lim2_idx = [info[1][1] for info in Fy_infos]
        Fy_cards = [info[2] for info in Fy_infos]

        Fy_lim1_patchgrid = self.get_2D_patchgrid(Fy_lim1_idx, suffix=hr_suffix, patched_shape=self.patched_shape[subdataset])
        Fy_lim2_patchgrid = self.get_2D_patchgrid(Fy_lim2_idx, suffix=hr_suffix, patched_shape=self.patched_shape[subdataset])

        nr,nc = Fy_lim1_patchgrid.shape[:2]
        
        base_image = self.get_full_img(subdataset, img_idx, hr_suffix)
        #base_image = np.zeros_like(base_image)
        Fy_lim1_img = torch.tensor(self.stitch_patches(np.array(Fy_lim1_patchgrid), patchsize=self.patchsizeX,grid_size=(nr,nc), base_image = np.array(base_image) ))
        Fy_lim2_img = torch.tensor(self.stitch_patches(np.array(Fy_lim2_patchgrid), patchsize=self.patchsizeX,grid_size=(nr,nc), base_image = np.array(base_image) ))

        # Fill the cardinal image : 
        cards = self.fill_patch_uniform_img(Fy_cards, patchsize=self.patchsizeX, patched_shape=self.patched_shape[subdataset])

        return Fy_lim1_img, Fy_lim2_img, cards

    def get_Fy_fullimg_idx_V2(self, subdataset, img_idx, feasible_info,feas_app, lim_area_ratio = 0.6, n_iter_max_ratio = 0.4):
        img_idx_list = self.imgs_idx_list[(subdataset, img_idx)]

        # For each index in the image, get the F_yidx info and put them into variables
        Fy_infos = [feasible_info[idx] for idx in img_idx_list]

        diams_Fy = torch.tensor([info[0] for info in Fy_infos])
        Fy_lim1_idx = torch.tensor([info[1][0] for info in Fy_infos])
        Fy_lim2_idx = torch.tensor([info[1][1] for info in Fy_infos])
        Fy_cards = [info[2] for info in Fy_infos]



        Fy_lim1_idx_small = self.fill_patch_info(Fy_lim1_idx, self.patched_shape[subdataset]) # indexes of the first patch corresponding to the diam of the local Fy, arranged in 2D
        Fy_lim2_idx_small = self.fill_patch_info(Fy_lim2_idx, self.patched_shape[subdataset]) # Same for the second patch corresponding to diam Fy

        diams_Fy_small = self.fill_patch_info(diams_Fy, self.patched_shape[subdataset])

        patched_shape = self.patched_shape[subdataset]
        # Sorting the diams_Fy keeping track of the positions in 2D
        flat_diams = diams_Fy_small.flatten()
        sorted_values, sorted_indices = torch.sort(flat_diams, descending=True)
        nc = patched_shape[1]
        rows = sorted_indices // nc
        cols = sorted_indices % nc
        indices_2d_diams = torch.stack((rows, cols), dim=1)
            


        replace_lim1_idx = np.zeros(patched_shape) # Stating which patch in the base image should be replaced by which one (in terms of indexes)
        replace_lim2_idx = np.zeros(patched_shape)

        replace_lim1_order = np.zeros(patched_shape)  # The order according to which the patches should be replaces
        replace_lim2_order = np.zeros(patched_shape)
        
        replaced_prop_lim1 = 0
        replaced_prop_lim2 = 0

        n_iter = 1  # number of iterations
        n_replace = 1 # number of times the patch replacement will be applied (smaller or equal to n_iter)
        
        nr,nc = patched_shape
        structuring_element = np.array([[0, 1, 0],
                                        [1, 1, 1],
                                        [0, 1, 0]], dtype=bool)

        while n_iter < nr*nc* n_iter_max_ratio and replaced_prop_lim1 < lim_area_ratio and replaced_prop_lim2 <lim_area_ratio : 
            maxdiam_position = indices_2d_diams[n_iter, :] #position of the patch verifying the max diam Fy, in the base image
            maxdiam_position = int(maxdiam_position[0]), int(maxdiam_position[1])
            i_base, j_base = maxdiam_position

            if False:
                maxdiam_1D_pos = self.get_patch_idx_img(maxdiam_position)
                print(f'Info Fy : {Fy_infos[maxdiam_1D_pos]}')


            idx_Fy_lim1 = Fy_lim1_idx_small[i_base, j_base] # index of the corresponding 1st patch (call it P1) associated to diam Fy 
            idx_Fy_lim2 = Fy_lim2_idx_small[i_base, j_base] # same for second one (call it P2)

            if False:
                print()
                print(f'idx_lim1 normallly : {idx_Fy_lim1}')
                print(f'idx_lim2 normallly : {idx_Fy_lim2}')

                print(f'Base position normally : {maxdiam_position}')
                base_1D_pos = self.get_patch_idx_img(maxdiam_position)
                base_idx = self.imgs_idx_list[(subdataset, img_idx)][base_1D_pos]
                print(f'Base idx normally : {base_idx}')
                print()

            if False:
                lim_patch1_id = self.patch_ids[int(idx_Fy_lim1)]
                img_lim1_id = self.get_img_id(int(idx_Fy_lim1))
                patch_lim1_nb = lim_patch1_id.split('_')[-1]
                
                position_img_lim1 = self.get_patch_position(int(patch_lim1_nb))
                i_lim1, j_lim1 = position_img_lim1

                pos1D_lim1 = self.get_patch_idx_img((i_lim1, j_lim1))

                idx_lim1 = self.imgs_idx_list[img_lim1_id][pos1D_lim1]

                

                print(f'idx_lim1 after f f^-1 :{idx_lim1}')
                print(f'img_lim1 : {img_lim1_id}')

                lim_patch2_id = self.patch_ids[int(idx_Fy_lim2)]
                img_lim2_id = self.get_img_id(int(idx_Fy_lim2))
                patch_lim2_nb = lim_patch2_id.split('_')[-1]
                position_img_lim2 = self.get_patch_position(int(patch_lim2_nb))
                i_lim2, j_lim2 = position_img_lim2

                pos1D_lim2 = self.get_patch_idx_img((i_lim2, j_lim2))

                idx_lim2 = self.imgs_idx_list[img_lim2_id][pos1D_lim2]

                print(f'idx_lim2 after f f^-1 :{idx_lim2}')
                print(f'img_lim2 : {img_lim2_id}')
                

            lim_patch1_id = self.patch_ids[int(idx_Fy_lim1)] # the corresponding patch ids
            lim_patch2_id = self.patch_ids[int(idx_Fy_lim2)]

            patched_shape_base = self.patched_shape[subdataset]

            img_lim1_id = self.get_img_id(int(idx_Fy_lim1)) # The id of the image where P1 belongs (subdataset, img idx)
            subds_img_lim1 = img_lim1_id[0]
            patched_shape_lim1 = self.patched_shape[subds_img_lim1]
            patch_lim1_nb = lim_patch1_id.split('_')[-1]  # Its place on the image, in the 1d array
            position_img_lim1 = self.get_patch_position(int(patch_lim1_nb), patched_shape= patched_shape_lim1 ) # its pplace on the image, but in 2D
            i_lim1, j_lim1 = position_img_lim1
            search_shift_lim1 = self.get_search_shift(maxdiam_position,position_img_lim1, patched_shape_base, patched_shape_lim1) # The area we search on around the position of p1 on its image
            search_area_lim1 = search_shift_lim1[0]+i_lim1,search_shift_lim1[1]+i_lim1, search_shift_lim1[2]+j_lim1, search_shift_lim1[3]+j_lim1 # This area + it position onthe image = area on the image
            same_feas_area_lim1, replacement_idx = self.get_same_feasible_area(maxdiam_position, position_img_lim1, search_shift_lim1, feas_app, (subdataset, img_idx), img_lim1_id,patched_shape_lim1, patched_shape_base )
            same_feas_area_lim1_shifted, replacement_idx_shifted_lim1 = self.shift_mask(same_feas_area_lim1, i_base-i_lim1, j_base-j_lim1, patched_shape_dst=patched_shape_base), self.shift_mask(replacement_idx, i_base-i_lim1, j_base-j_lim1, patched_shape_dst=patched_shape_base)

            img_lim2_id = self.get_img_id(int(idx_Fy_lim2))
            subds_img_lim2 = img_lim2_id[0]
            patched_shape_lim2 = self.patched_shape[subds_img_lim2]
            patch_lim2_nb = lim_patch2_id.split('_')[-1]
            position_img_lim2 = self.get_patch_position(int(patch_lim2_nb), patched_shape=patched_shape_lim2)
            i_lim2, j_lim2 = position_img_lim2
            search_shift_lim2 = self.get_search_shift(maxdiam_position,position_img_lim2, patched_shape_base, patched_shape_lim2)
            search_area_lim2 = search_shift_lim2[0]+i_lim2,search_shift_lim2[1]+i_lim2, search_shift_lim2[2]+j_lim2, search_shift_lim2[3]+j_lim2
            same_feas_area_lim2, replacement_idx = self.get_same_feasible_area(maxdiam_position, position_img_lim2, search_shift_lim2, feas_app, (subdataset, img_idx), img_lim2_id, patched_shape_lim2, patched_shape_base)
            same_feas_area_lim2_shifted, replacement_idx_shifted_lim2 = self.shift_mask(same_feas_area_lim2, i_base-i_lim2, j_base-j_lim2, patched_shape_dst=patched_shape_base), self.shift_mask(replacement_idx, i_base-i_lim2, j_base-j_lim2, patched_shape_dst=patched_shape_base)


            # Verify whether there is an overlap between the dilation of the iterated mask and the mask for this mask
            iterative_mask_lim1 = replace_lim1_idx> 0.5
            iterative_mask_lim2 = replace_lim2_idx> 0.5

            dilated_iterative_mask_lim1 = binary_dilation(iterative_mask_lim1, structure=structuring_element)
            dilated_iterative_mask_lim2 = binary_dilation(iterative_mask_lim2, structure=structuring_element)

            intersection_lim1 = np.logical_and(same_feas_area_lim1_shifted,dilated_iterative_mask_lim1 )
            intersection_lim2 = np.logical_and(same_feas_area_lim2_shifted,dilated_iterative_mask_lim2 )

            if not np.any(intersection_lim1) and not np.any(intersection_lim2):
                # Fill the iterated replacement patch indexes
                replace_lim1_idx = np.where(same_feas_area_lim1_shifted >0.5, replacement_idx_shifted_lim1, replace_lim1_idx)
                replace_lim2_idx = np.where(same_feas_area_lim2_shifted >0.5, replacement_idx_shifted_lim2, replace_lim2_idx)
                
                # Fill the order of filling in the iterated orders of fillings
                replace_lim1_order = np.where(same_feas_area_lim1_shifted >0.5, n_replace, replace_lim1_order)
                replace_lim2_order = np.where(same_feas_area_lim2_shifted >0.5, n_replace, replace_lim2_order)


                n_replace +=1



            n_iter +=1

            closed_mask_lim1 = binary_closing(replace_lim1_order>0.5, structure = structuring_element)
            closed_mask_lim2 = binary_closing(replace_lim2_order>0.5, structure = structuring_element)


            replaced_prop_lim1 = np.sum(closed_mask_lim1>0.5)/(patched_shape[0] * patched_shape[1])
            replaced_prop_lim2 = np.sum(closed_mask_lim2>0.5)/(patched_shape[0] * patched_shape[1])

            #print(f'n_iter = {n_iter-1}     n_replace = {n_replace-1} \n replaced_prop_lim1 = {np.round(100*replaced_prop_lim1)}%    replaced_prop_lim2 = {np.round(100*replaced_prop_lim2)}%')

            if False and n_iter %100 ==0:
           
                fig, axs = plt.subplots(1, 2, figsize=(10, 10))
                axs = axs.flatten()

             
                axs[0].imshow(replace_lim1_order>0.5, cmap='gray')
                axs[0].set_title('Replaced areas lim 1 ')
                axs[1].imshow(replace_lim2_order>0.5, cmap='gray')
                axs[1].set_title('Replaced areas lim 2')

                plt.tight_layout()
                plt.show()
                pass

             # For eqch limit image, 
             # 1. Get the area of search using position in the imglim 

             #Within the condition that the patch is not in the masked area
             # 2.Check for which patches, the patch if imglim belongs to the corresponding feasible set in the base image (shift to take in account)        mask_lim1 = torch.zeros(self.patched_shape)
             # 3. Take the connected component around the patch in question and paste it to the basee image
        
        return replace_lim1_idx,replace_lim2_idx, replace_lim1_order, replace_lim2_order

    def get_Fy_fullimg_V2(self, subdataset, img_idx, feasible_info,feas_app, hr_suffix = 'hr_res', lr_suffix = 'lr_res',  lim_area_ratio = 0.6, n_iter_max_ratio = 0.4, sigma_blend = 1, margin_blend = 1):

        replace_lim1_idx,replace_lim2_idx, replace_lim1_order, replace_lim2_order = self. get_Fy_fullimg_idx_V2(subdataset = subdataset, img_idx = img_idx, feasible_info = feasible_info, feas_app= feas_app, lim_area_ratio=lim_area_ratio, n_iter_max_ratio=n_iter_max_ratio)
        replace_lim1_idx_flat,replace_lim2_idx_flat, replace_lim1_order_flat, replace_lim2_order_flat = replace_lim1_idx.flatten(),replace_lim2_idx.flatten(), replace_lim1_order.flatten(), replace_lim2_order.flatten()
        # Get the indexlist of the original image
        baseimg_idx_list = np.array(self.imgs_idx_list[(subdataset, img_idx)])

        # For v1 : fill the image directly with the index list
        # For v2 : use patch blending to fill effectively the lim images
        blending = True
        if not blending:
            # Fill the indexes arrays by replacing
            img_lim1_idx_flat = np.where(replace_lim1_order_flat>0.5,replace_lim1_idx_flat,baseimg_idx_list ).astype(int)
            img_lim2_idx_flat = np.where(replace_lim2_order_flat>0.5,replace_lim2_idx_flat,baseimg_idx_list ).astype(int)

            Fy_lim1_img = self.recompose_image(img_lim1_idx_flat, suffix = hr_suffix, patched_shape= self.patched_shape[subdataset])
            Fy_lim2_img = self.recompose_image(img_lim2_idx_flat, suffix= hr_suffix, patched_shape= self.patched_shape[subdataset])

            Fy_lim1_imgY = self.recompose_image(img_lim1_idx_flat, suffix = lr_suffix, patched_shape= self.patched_shape[subdataset])
            Fy_lim2_imgY = self.recompose_image(img_lim2_idx_flat, suffix= lr_suffix, patched_shape= self.patched_shape[subdataset])
        else:
            base_imgX = self.recompose_image(baseimg_idx_list, suffix=hr_suffix, patched_shape= self.patched_shape[subdataset])
            img_lim1_idx_flat = np.where(replace_lim1_order_flat>0.5,replace_lim1_idx_flat,baseimg_idx_list ).astype(int)
            img_lim2_idx_flat = np.where(replace_lim2_order_flat>0.5,replace_lim2_idx_flat,baseimg_idx_list ).astype(int)

            Fy_lim1_imgY = self.recompose_image(img_lim1_idx_flat, suffix = lr_suffix, patched_shape= self.patched_shape[subdataset])
            Fy_lim2_imgY = self.recompose_image(img_lim2_idx_flat, suffix= lr_suffix, patched_shape= self.patched_shape[subdataset])


            # Isolate the indexes corresponding to that order in the 2D array
            res_mask_lim1 = replace_lim1_order >0.5
            res_bigmask_lim1 = self.amplify_mask(res_mask_lim1, size = 'big', patched_shape=self.patched_shape[subdataset]) # This is the mask telling where the patches will be sticked
            

            # Erode the blurred mask by 1 pixel
            kernel = np.ones((3, 3), np.uint8)
            eroded_mask = cv2.erode(res_bigmask_lim1, kernel, iterations=1)
            # Gaussian convolution of the mask 
            blurred_mask = torch.tensor(gaussian_filter(eroded_mask, sigma = sigma_blend))
            


            Fy_lim1_img_noblend = self.recompose_image(img_lim1_idx_flat, suffix = hr_suffix, patched_shape = self.patched_shape[subdataset])
            Fy_lim1_img = blurred_mask * Fy_lim1_img_noblend + (1-blurred_mask)*base_imgX


            # Isolate the indexes corresponding to that order in the 2D array
            res_mask_lim2 = replace_lim2_order >0.5
            res_bigmask_lim2 = self.amplify_mask(res_mask_lim2, size = 'big', patched_shape=self.patched_shape[subdataset]) # This is the mask telling where the patches will be sticked

            # Erode the blurred mask by 1 pixel
            eroded_mask = cv2.erode(res_bigmask_lim2, kernel, iterations=1)
            # Gaussian convolution of the mask
            blurred_mask = torch.tensor(gaussian_filter(eroded_mask, sigma = sigma_blend))

            Fy_lim2_img_noblend = self.recompose_image(img_lim2_idx_flat, suffix = hr_suffix, patched_shape= self.patched_shape[subdataset])
            Fy_lim2_img = blurred_mask * Fy_lim2_img_noblend + (1-blurred_mask)*base_imgX

        return Fy_lim1_img, Fy_lim2_img, Fy_lim1_imgY, Fy_lim2_imgY
    
    
    def find_bounding_rectangle_mask(self, mask):
        rows, cols = np.where(mask >0.5)
        if len(rows) > 0 and len(cols) > 0:
            # Bounding box: min/max row and column
            ymin = np.min(rows)
            xmin = np.min(cols)
            ymax = np.max(rows)+1
            xmax = np.max(cols)+1
            return ymin, ymax, xmin, xmax
        else:
            print("No non-zero pixels found.")
            return 0,0,0,0
        
    
    def amplify_mask(self, small_mask, size,patched_shape):
        nr,nc = patched_shape
        if size == 'big':
            ps = self.patchsizeX
        elif size == 'small':
            ps = self.patchsizeY
        else:
            print('Please choose a size among big or small')
            raise ValueError
        big_mask = np.zeros((nr*ps, nc*ps))

        for i in range(nr):
            for j in range(nc):
                big_mask[i*ps: (i+1)*ps, j*ps: (j+1)*ps] = small_mask[i,j]
        return big_mask

    
    def get_same_feasible_area(self, position_base, position_lim, search_shift_lim, feas_app, base_img_id, lim_img_id, patched_shape_lim, patched_shape_base):
        '''
        Given the area to search in img_lim, for each patch in this img_lim, 
        we check whether it belongd in the feasible of the corresponding Fy of the base img
        '''
        i_lim, j_lim = position_lim
        i_base, j_base = position_base

        mask_same_feasible_lim = np.zeros(patched_shape_lim)
        replacement_idx = np.zeros(patched_shape_lim)
        sh_ymin, sh_ymax, sh_xmin, sh_xmax = search_shift_lim

        for i in range(sh_ymin, sh_ymax):
            for j in range(sh_xmin, sh_xmax):
                pos_1D_base = self.get_patch_idx_img((i+ i_base, j+j_base), patched_shape_base)
                idx_base = self.imgs_idx_list[base_img_id][pos_1D_base]

                pos1D_lim = self.get_patch_idx_img((i+i_lim, j+j_lim), patched_shape_lim)
                idx_lim = self.imgs_idx_list[lim_img_id][pos1D_lim]
            
                mask_same_feasible_lim[i+i_lim, j+j_lim] = feas_app[idx_lim, idx_base]
                if mask_same_feasible_lim[i+i_lim, j+j_lim] >0.5:
                    replacement_idx[i+i_lim, j+j_lim] = idx_lim


                if i==0 and j==0 and feas_app[idx_lim, idx_base] ==0 and True:
                    
                    print()
                    patch_id_base = self.patch_ids[idx_base]

                    img_base_id = self.get_img_id(int(idx_base))
                    subds_img_base = img_base_id[0]

                    position_1D_base = int(patch_id_base.split('_')[-1])
                    position_2D_base = self.get_patch_position(position_1D_base, self.patched_shape[subds_img_base])
                    print('Unexpected behaviour in the image reconstitution')
                    print(f'Looked Base position : {position_2D_base}')
                    print(f'Looked indexes : {idx_lim}, {idx_base}')


        labeled_array, num_features = label(mask_same_feasible_lim)
        component_label = labeled_array[i_lim, j_lim]
   

        if component_label == 0:
            return  np.zeros_like(mask_same_feasible_lim, dtype=bool), np.zeros_like(mask_same_feasible_lim)
        

        mask_same_feasible_lim = labeled_array == component_label
        replacement_idx = replacement_idx * mask_same_feasible_lim
        return mask_same_feasible_lim, replacement_idx
        
        
    def get_search_shift(self,position_base, position_lim, patched_shape_base, patched_shape_lim):
        i_base, j_base = position_base
        i_lim, j_lim = position_lim
        nr_base,nc_base = patched_shape_base
        nr_lim,nc_lim = patched_shape_lim


        y_min , y_max= -min(i_base, i_lim), min(nr_base-i_base, nr_lim-i_lim)
        x_min, x_max = -min(j_base, j_lim), min(nc_base-j_base, nc_lim-j_lim)
        return y_min, y_max, x_min, x_max


    def shift_mask(self,mask, k, l, patched_shape_dst):
        shifted_res = np.zeros_like(mask)
        
        rows, cols = mask.shape

        nr_dst, nc_dst = patched_shape_dst
        
        # Calculate the source and destination slice ranges
        src_row_start = max(0, -k)
        src_row_end = rows - max(0, k)
        
        src_col_start = max(0, -l)
        src_col_end = cols - max(0, l)
        
        dst_row_start = max(0, k)
        dst_row_end = rows - max(0, -k)
        
        dst_col_start = max(0, l)
        dst_col_end = cols - max(0, -l)
        
        shifted_res[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = mask[src_row_start:src_row_end, src_col_start:src_col_end]
        if rows <= nr_dst: # destination bigger than source
            shifted = np.zeros(patched_shape_dst)
            shifted[:rows, :cols] = shifted_res
        else: # destination smaller than source
            shifted = shifted_res[:nr_dst, nc_dst]
        
        return shifted

    def get_Fy_lim_fullimg_V3(self, subdataset, img_idx, feasible_info, feas_app,sigma_blend = 1, hr_suffix = 'hr_res', lr_suffix = 'lr_res'):
        img_idx_list = self.imgs_idx_list[(subdataset, img_idx)]

        Fy_infos = [feasible_info[idx] for idx in img_idx_list]
        Fy_lim1_idx = [info[1][0] for info in Fy_infos]
        Fy_lim2_idx = [info[1][1] for info in Fy_infos]

        # Get the full base image as initialization
        base_imgX = self.recompose_image(img_idx_list, suffix=hr_suffix, patched_shape= self.patched_shape[subdataset])

        Fy_lim1_Y = self.recompose_image(Fy_lim1_idx, suffix = lr_suffix, patched_shape= self.patched_shape[subdataset])
        Fy_lim2_Y = self.recompose_image(Fy_lim2_idx, suffix = lr_suffix, patched_shape= self.patched_shape[subdataset])

        res_lim1 = np.copy(base_imgX.numpy())
        res_lim2 = np.copy(base_imgX.numpy())

        c,h,w = base_imgX.shape
        kernel = np.ones((3, 3), np.uint8)
        # For each patch, stick it to res with alpha blending (after erosion and blurring of the mask for the patch)
        n_pat = len(Fy_lim1_idx)
        psX = self.patchsizeX
        for k in range(n_pat):
            i,j = self.get_patch_position(k, self.patched_shape[subdataset])
            mask_patch = np.zeros((h,w))
            mask_patch[i*psX:(i+1)*psX, j*psX:(j+1)*psX] = 1
            # ERODE THE MASK AND BLUR IT
            eroded_mask = cv2.erode(mask_patch, kernel, iterations=1)
            blurred_mask = torch.tensor(gaussian_filter(eroded_mask, sigma = sigma_blend))

            patch_lim1 = self.__getitem__(Fy_lim1_idx[k])[hr_suffix]
            patch_lim1_onimg = np.zeros_like(base_imgX)
            patch_lim1_onimg[:,i*psX:(i+1)*psX, j*psX:(j+1)*psX] = patch_lim1.numpy()
            # Alpha blending for the patch
            res_lim1 = blurred_mask * patch_lim1_onimg + (1-blurred_mask)*res_lim1

            patch_lim2 = self.__getitem__(Fy_lim2_idx[k])[hr_suffix]
            patch_lim2_onimg = np.zeros_like(base_imgX)
            patch_lim2_onimg[:,i*psX:(i+1)*psX, j*psX:(j+1)*psX] = patch_lim2.numpy()
            # Alpha blending for the patch
            res_lim2 = blurred_mask * patch_lim2_onimg + (1-blurred_mask)*res_lim2
        return res_lim1, res_lim2, Fy_lim1_Y, Fy_lim2_Y



    def get_Fy_lim_fullimg(self, subdataset, img_idx, feasible_info, hr_suffix = 'hr_res', lr_suffix = 'lr_res'):

        img_idx_list = self.imgs_idx_list[(subdataset, img_idx)]
        # Compose the images per patch : 2 images corresponfing to the 2 that achieve diam(F_y) and the card of the F_y

        # For each index in the image, get the F_yidx info and put them into variables
        Fy_infos = [feasible_info[idx] for idx in img_idx_list]

        diams_Fy = [info[0] for info in Fy_infos]
        Fy_lim1_idx = [info[1][0] for info in Fy_infos]
        Fy_lim2_idx = [info[1][1] for info in Fy_infos]
        Fy_cards = [info[2] for info in Fy_infos]


        # compose the images at the borders of F_y
        Fy_lim1_img = self.recompose_image(Fy_lim1_idx, suffix=hr_suffix, patched_shape= self.patched_shape[subdataset])
        Fy_lim2_img = self.recompose_image(Fy_lim2_idx, suffix=hr_suffix, patched_shape= self.patched_shape[subdataset])


        Fy_lim1_imgY = self.recompose_image(Fy_lim1_idx, suffix = lr_suffix, patched_shape= self.patched_shape[subdataset])
        Fy_lim2_imgY = self.recompose_image(Fy_lim2_idx, suffix = lr_suffix, patched_shape= self.patched_shape[subdataset])

        # Fill the cardinal image : 
        cards = self.fill_patch_uniform_img(Fy_cards, patchsize=self.patchsizeX, patched_shape=self.patched_shape[subdataset])


        return Fy_lim1_img, Fy_lim2_img,Fy_lim1_imgY, Fy_lim2_imgY, cards



    def get_full_img(self,subdataset, img_idx, suffix):
        
        
        img_idx_list = self.imgs_idx_list[(subdataset, str(img_idx))]
        return self.recompose_image(img_idx_list, suffix, patched_shape= self.patched_shape[subdataset])



if __name__ =='__main__':


    parser = argparse.ArgumentParser(description="Patched dataset creation Script")
    parser.add_argument('--full_img_datapath',type = str, default='/p/project1/hai_1013/sat_data/cross_processed' ,help='The folder you store your fulll images in')
    parser.add_argument('--out_folder',type = str, default = '/p/project1/hai_1013/sat_data/patched_crossproc',help='Where do you want to save your patched dataset')
    parser.add_argument('--DSHR', action = 'store_true', default=True, help='If activated, the LR image will be DS(HR) with DS beint the torch Ds operator')
    parser.add_argument('--SR_factor',type = int ,default = 4,help='The scale factor you want to use in case you want the DSHR dataset')

    args = parser.parse_args()

    full_img_datapath = args.full_img_datapath
    out_folder = args.out_folder
    DSHR = args.DSHR
    SR_factor = args.SR_factor

    SR_factor = 4

    if DSHR:
        out_folder = f'{out_folder}_DSHR{SR_factor}'

    patchsizesX = [8,12,16,20]
    
    subdatasets = ['naip', 'spot', 'spain_crops', 'spain_urban']

    for PS_X in patchsizesX:
        print(f'Creating patched dataset for a patchsize of {PS_X}')
        if not DSHR:
            build_S2_patched_dataset(patchsize_X = PS_X,img_dset_folder =  full_img_datapath, subdatasets  = subdatasets, out_dsfolder =  os.path.join(out_folder, f'PS{PS_X}'), labels = ('hr_data', 'lr_data'), border_X = 0, SR_factor = SR_factor)
        else:
            build_S2_patched_dataset_DSHR(patchsize_X = PS_X, img_dset_folder= full_img_datapath, subdatasets= subdatasets,out_dsfolder= os.path.join(out_folder, f'PS{PS_X}') ,labels = ('hr_data', 'lr_data'), border_X = 0, SR_factor = SR_factor )

        print('Done')

            
