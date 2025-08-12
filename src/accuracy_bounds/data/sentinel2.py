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

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        patch_id = self.patch_ids[idx]
        
        if len(self.suffix_list) == 1: 
            img_path = os.path.join(self.folder_path, f'{patch_id}_{self.suffix_list[0]}.tif')

            with rasterio.open(lr_path) as src:
                img = torch.from_numpy(src.read())
            
            return {
                    'name': patch_id,
                    'img': img
                }
        else:
            result_dict = {"name": patch_id}

            for suffix in self.suffix_list: 

                img_path = os.path.join(self.folder_path, f'{patch_id}_{suffix}.tif')

                with rasterio.open(lr_path) as src:
                    result_dict["suffix"] = torch.from_numpy(src.read())

            return result_dict

            
