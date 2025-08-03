import os
import glob
import torch
from torch.utils.data import Dataset
import rasterio

class SRDataset(Dataset):
    def __init__(self, folder_path, suffixes=('lr', 'sr', 'hr')):
        """
        Args:
            folder_path (str): Path to folder containing image patches.
            suffixes (tuple): Suffixes used in filenames for LR, SR, HR.
        """
        self.folder_path = folder_path
        self.lr_suffix, self.sr_suffix, self.hr_suffix = suffixes

        # Gather all _lr.tif files and derive basename (e.g., 0001_0001)
        self.lr_files = sorted(glob.glob(os.path.join(folder_path, f'*_{self.lr_suffix}.tif')))
        self.patch_ids = [os.path.basename(f).replace(f'_{self.lr_suffix}.tif', '') for f in self.lr_files]

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        patch_id = self.patch_ids[idx]
        
        lr_path = os.path.join(self.folder_path, f'{patch_id}_{self.lr_suffix}.tif')
        assert lr_path == self.lr_files[idx]
        
        with rasterio.open(lr_path) as src:
            lr = torch.from_numpy(src.read())

        sr_path = os.path.join(self.folder_path, f'{patch_id}_{self.sr_suffix}.tif')
        with rasterio.open(sr_path) as src:
            sr = torch.from_numpy(src.read())

        hr_path = os.path.join(self.folder_path, f'{patch_id}_{self.hr_suffix}.tif')
        with rasterio.open(hr_path) as src:
            hr = torch.from_numpy(src.read())

        return {
            'name': patch_id,
            'lr': lr,
            'sr': sr,
            'hr': hr
        }
