import os
import glob
import torch
from torch.utils.data import Dataset
import rasterio

class SRDataset(Dataset):
    def __init__(self, folder_path, suffixes=('lr', 'hr')):
        """
        Args:
            folder_path (str): Path to folder containing image patches.
            suffixes (tuple): Suffixes used in filenames for LR, HR.
        """

        self.folder_path = folder_path
        self.suffix_list = suffixes if not isinstance(suffixes, str) else [suffixes]

        # Gather all _lr.tif files and derive basename (e.g., 0001_0001)
        self.file_list = sorted(glob.glob(os.path.join(folder_path, f'*_{self.suffix_list[0]}.tif')))
        self.patch_ids = [os.path.basename(f).replace(f'_{self.suffix_list[0]}.tif', '') for f in self.file_list]
        self.image_size = (39*12, 39*12)
        

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
        
    def recompose_image(self,idx_list, suffix):
        assert suffix in self.suffix_list

        h,w = self.image_size


        patches_id = [self.patch_ids[idx] for idx in idx_list]
        paths = [os.path.join(self.folder_path, f'{patch_id}_{suffix}.tif') for patch_id in patches_id]
        patches = []
        for path in paths:
            with rasterio.open(path) as src:
                patches.append(torch.from_numpy(src.read()))

        patches = torch.stack(patches) 
        c = patches.shape[1]
        ps = patches.shape[2]



        n_patches_y = h//ps
        n_patches_x = w//ps

        patches = patches.view(n_patches_y, n_patches_x, c, ps, ps)
        reconstructed = patches.permute(2, 0, 3, 1, 4)
        reconstructed = reconstructed.reshape(c, n_patches_y * ps, n_patches_x * ps)

        return reconstructed



    def get_full_img(self,subdataset, img_idx, suffix):
        img_idx_list = [(i, self.patch_ids[i].split('_')[2]) for i in range(len(self.patch_ids)) if (f'{subdataset}_{img_idx}' in self.patch_ids[i]) ]
        


        img_idx_list = sorted(
            img_idx_list,
            key=lambda x: int(x[1])
        )

        img_idx_list = [int(x[0]) for x in img_idx_list]
        

        return self.recompose_image(img_idx_list, suffix)


            
