import torch
from torch.utils.data import DataLoader
from accuracy_bounds.inverseproblems.utils import feasibleApp_samplingYX_perbatch_cuda,target_distances_samplingYX_precomputedFA_cuda_V2, target_distances_samplingYX_precomputedFA_perbatch_cuda
import numpy as np
import time

class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, data_array, suffix):
        if suffix == 'x':
            self.data = data_array
        elif suffix == 'y':
            self.data = data_array[:,:2]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'idx': idx, 'img': self.data[idx]}
    

# Parameters
n = 10000  # Number of vectors in dataset
epsilon = 0.039  # Distance threshold for similarity
batch_size = 1000  # Batch size for DataLoader

data_3D = torch.randn(n, 3).float()
data_3D = data_3D / data_3D.norm(dim=1, keepdim=True) # Random points on the unit sphere

# Create datasets and dataloaders for dataset1 and dataset2
dataset_3D = RandomDataset(data_array= data_3D,suffix='x')
dataset_proj = RandomDataset(data_array=data_3D, suffix='y')


dataloader_3D = DataLoader(dataset_3D, batch_size=batch_size, shuffle=False)
dataloader_3D_2 = DataLoader(dataset_3D, batch_size=batch_size, shuffle=False)
dataloader_proj = DataLoader(dataset_proj, batch_size=batch_size, shuffle=False)
dataloader_forwarded3D = DataLoader(dataset_proj, batch_size=batch_size, shuffle=False)

feas_app = feasibleApp_samplingYX_perbatch_cuda(0, dataloader_proj, dataloader_forwarded3D, p_Y=2, epsilon=epsilon)
t0 = time.time()
distsXX_new, feas_app_new = target_distances_samplingYX_precomputedFA_cuda_V2(0, dataloader_3D, feas_app, p_X = 1, batchsize=10000)
t1 = time.time()
distsXX_old, feas_app_old = target_distances_samplingYX_precomputedFA_perbatch_cuda(0,dataloader_3D, dataloader_3D_2, feas_app, p_X = 1)
t2 = time.time()
print(f'Time taken for the new method= {np.round(t1-t0, 3)} seconds')
print(f'Time taken for the old method= {np.round(t2-t1, 3)} seconds')

h,w = feas_app.shape
print(f'Sparsity ratio of feasible apprtenance = {feas_app.sum()/(h*w)}')
print(torch.norm(distsXX_old.to_dense())**2)  
print(torch.norm(distsXX_new.to_dense())**2)  
