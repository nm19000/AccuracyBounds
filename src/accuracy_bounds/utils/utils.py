import numpy as np
import random
import torch 
import rasterio
from rasterio.transform import from_origin
import os
from tqdm import tqdm


# Function to apply matrix transformation A to points
def apply_forwardmodel(A, points):
    return np.dot(points, A.T)

def projection_nullspace_operator(A):
    """Compute the matrix for projecting onto the null space of a matrix A, i.e. P_{N(A)}= (I - A^dagger A)
    Args: 
        - A: matrix 
    Returns:
        - project_ns: matrix projecting onto the null space of A.
    """
    A_dagger = np.linalg.pinv(A)
    project_ns= np.eye(A.shape[1]) - np.dot(A_dagger, A)
    return project_ns


def projection_nullspace(A, x):
    """
        Compute the projection of a point x onto the null space of A, i.e., P_{N(A)}(x).
        This is equivalent to (I - A^dagger A) x usin the function projection_nullspace for computing P_{N(A)} from A.
    """
    project_ns = projection_nullspace_operator(A)
    x_ns = np.dot(project_ns,x)
    
    return x_ns


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_into_tiff(bands, out_path):
    # Save bands as a TIFF file
    with rasterio.open(
                out_path,
                'w',
                driver='GTiff',
                height=bands.shape[1],
                width=bands.shape[2],
                count=bands.shape[0],
                dtype='float32',
                crs='+proj=latlong',
                transform=from_origin(0, 0, 10, 10)  # Example transform, adjust as needed
                ) as dst:
                for i in range(bands.shape[0]):
                        dst.write(bands[i, :, :].astype(np.float32), i + 1)

    pass

def get_patches_from_S2(img,patchsize, border):
    h,w = img.shape[1:]
    n_patches_y = (h-border-patchsize)//patchsize
    n_patches_x = (w-border-patchsize)//patchsize
    all_patches = []
    for i in range(n_patches_y):
        imin = border + i*patchsize
        imax = imin + patchsize
        for j in range(n_patches_x):
            jmin = border + j*patchsize
            jmax = jmin + patchsize
            all_patches.append(img[:, imin:imax, jmin:jmax])
    
    all_patches = torch.stack(all_patches)
    return all_patches

def build_S2_patched_dataset(patchsize_X,img_dset_folder , subdatasets , out_dsfolder, labels = ('hr_data', 'lr_data'), border_X = 0, SR_factor = 4):
    border_Y = border_X//SR_factor
    patchsize_Y = patchsize_X//SR_factor

    hr_label = labels[0]
    lr_label = labels[1]


    for subds in subdatasets:
        data_folder = os.path.join(img_dset_folder, subds)
        img_folders = os.listdir(data_folder)
        bar = [x for x in img_folders if 'json' not in x]

        for idxstr in tqdm(bar):
            img_folder = os.path.join(data_folder, idxstr)
            lr_path = f'{img_folder}/{lr_label}.tif'
            hr_path = f'{img_folder}/{hr_label}.tif'

            with rasterio.open(hr_path) as hr_src:
                hr_img = hr_src.read()
            hr_img = torch.from_numpy(hr_img) 
            with rasterio.open(lr_path) as lr_src:
                lr_img = lr_src.read()
            lr_img = torch.from_numpy(lr_img) 

            #metrics.setup(lr=lr_img, sr=sr_img, hr=hr_img)

            patched_lr = get_patches_from_S2(lr_img, patchsize=patchsize_Y, stride=patchsize_Y, border=border_Y)
            patched_hr = get_patches_from_S2(hr_img, patchsize_X,patchsize_X, border_X)

            m = patched_lr.shape[0]
            for i in range(m):
                save_into_tiff(bands=np.array(patched_lr[i]), out_path=os.path.join(out_dsfolder, f'{subds}_{idxstr}_{i}_{lr_label}.tif'))
                save_into_tiff(bands=np.array(patched_hr[i]), out_path=os.path.join(out_dsfolder, f'{subds}_{idxstr}_{i}_{hr_label}.tif'))

