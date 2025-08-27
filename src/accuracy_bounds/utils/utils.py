import numpy as np
import random
import torch 
import rasterio
from rasterio.transform import from_origin
import os
from tqdm import tqdm
from accuracy_bounds.inverseproblems.utils import apply_upsampling
from typing import List, Optional, Union
from abc import ABC, abstractmethod


class DistanceMetric(ABC):
    """An abstract class to compute the distance between two tensors.

    Parameters:
        method (str): The method to use. Either "pixel", "patch", or "image".
        patch_size (int): The patch size to use if the patch method is used.
        x (torch.Tensor): The SR harmonized image (C, H, W).
        y (torch.Tensor): The HR image (C, H, W).
        **kwargs: The parameters to pass to the distance function.

    Abstract methods:
        compute_patch: Compute the distance metric at patch level.
        compute_image: Compute the distance metric at image level.
        compute_pixel: Compute the disDLR VPNtance metric at image level.
        compute: Compute the distance metric.
    """

    def __init__(
        self, method: str, patch_size: int, x: torch.Tensor, y: torch.Tensor, **kwargs
    ):
        self.method = method
        self.patch_size = patch_size
        self.kwargs = kwargs
        self.axis: int = 0
        self.x = x
        self.y = y

    @staticmethod
    def do_square(tensor: torch.Tensor, patch_size: Optional[int] = 32) -> torch.Tensor:
        """Split a tensor into n_patches x n_patches patches and return
        the patches as a tensor.

        Args:
            tensor (torch.Tensor): The tensor to split.
            n_patches (int, optional): The number of patches to split the tensor into.
                If None, the tensor is split into the smallest number of patches.

        Returns:
            torch.Tensor: The patches as a tensor.
        """

        # Check if it is a square tensor
        if tensor.shape[-1] != tensor.shape[-2]:
            raise ValueError("The tensor must be square.")

        # Get the image size
        xdim = tensor.shape[1]
        ydim = tensor.shape[2]

        # Get the patch size
        minimages_x = int(torch.ceil(torch.tensor(xdim / patch_size)))
        minimages_y = int(torch.ceil(torch.tensor(ydim / patch_size)))

        # pad the tensor to be divisible by the patch size
        pad_x_01 = int((minimages_x * patch_size - xdim) // 2)
        pad_x_02 = int((minimages_x * patch_size - xdim) - pad_x_01)

        pad_y_01 = int((minimages_y * patch_size - ydim) // 2)
        pad_y_02 = int((minimages_y * patch_size - ydim) - pad_y_01)

        padded_tensor = torch.nn.functional.pad(
            tensor, (pad_x_01, pad_x_02, pad_y_01, pad_y_02)
        )

        # split the tensor (C, H, W) into (n_patches, n_patches, C, H, W)
        patches = padded_tensor.unfold(1, patch_size, patch_size).unfold(
            2, patch_size, patch_size
        )

        # move the axes (C, n_patches, n_patches, H, W) -> (n_patches, n_patches, C, H, W)
        patches = patches.permute(1, 2, 0, 3, 4)

        return patches

    @abstractmethod
    def _compute_image(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    def compute_image(self) -> torch.Tensor:
        return self._compute_image(self.x, self.y)

    def compute_patch(self) -> torch.Tensor:
        # Create the patches
        x_batched = self.do_square(self.x, self.patch_size)
        y_batched = self.do_square(self.y, self.patch_size)

        # Compute the metric for each patch
        metric_result = torch.zeros(x_batched.shape[:2])
        xrange, yrange = x_batched.shape[0:2]
        for x_index in range(xrange):
            for y_index in range(yrange):
                x_batch = x_batched[x_index, y_index]
                y_batch = y_batched[x_index, y_index]
                metric_result[x_index, y_index] = self._compute_image(x_batch, y_batch)

        # Go back to the original size
        metric_result = torch.nn.functional.interpolate(
            metric_result[None, None], size=self.x.shape[-2:], mode="nearest"
        ).squeeze()

        return metric_result

    def compute_pixel(self) -> torch.Tensor:
        return self._compute_pixel(self.x, self.y)

    def compute(self) -> torch.Tensor:
        if self.method == "pixel":
            return self.compute_pixel()
        elif self.method == "image":
            return self.compute_image()
        elif self.method == "patch":
            return self.compute_patch()
        else:
            raise ValueError("Invalid method.")


class L1(DistanceMetric):
    """Spectral information divergence between two tensors"""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        method: str = "image",
        patch_size: int = 32,
    ):
        super().__init__(x=x, y=y, method=method, patch_size=patch_size)

    def _compute_image(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nanmean(torch.abs(x - y))

    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nanmean(torch.abs(x - y), axis=0)


class L2(DistanceMetric):
    """Spectral information divergence between two tensors"""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        method: str = "image",
        patch_size: int = 32,
    ):
        super().__init__(x=x, y=y, method=method, patch_size=patch_size)

    def _compute_image(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nanmean((x - y) ** 2)

    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nanmean((x - y) ** 2, axis=0)

class LP(DistanceMetric):
    """Spectral information divergence between two tensors"""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        p,
        method: str = "image",
        patch_size: int = 32,
    ):
        self.p = p
        super().__init__(x=x, y=y, method=method, patch_size=patch_size)

    def _compute_image(self, x:torch.Tensor, y:torch.Tensor)-> torch.Tensor:
        return (torch.nanmean(torch.abs(x - y) ** self.p))**(1/self.p)
    
    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (torch.nanmean(torch.abs(x - y) ** self.p))**(1/self.p)
  

class Struct(DistanceMetric):
    '''
    Structure term in the SSIM (without the constants c1,2)
    '''

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        method: str = "image",
        patch_size: int = 32,
    ):

        super().__init__(x=x, y=y, method=method, patch_size= patch_size)

    def _compute_image(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        x_center = x - torch.mean(x)
        y_center = y - torch.mean(y)

        dot_product = (x_center * y_center).squeeze().sum()
        preds_norm = x_center.squeeze().norm()
        target_norm = y_center.squeeze().norm()
        sam_score = torch.clamp(dot_product / (preds_norm * target_norm), -1, 1).acos()
        return torch.rad2deg(sam_score)

    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_center = x - torch.mean(x)
        y_center = y - torch.mean(y)
        dot_product = (x_center * y_center).sum(dim=0)
        preds_norm = x_center.norm(dim=0)
        target_norm = y_center.norm(dim=0)
        sam_score = torch.clamp(dot_product / (preds_norm * target_norm), -1, 1).acos()
        return torch.rad2deg(sam_score)

def depatchify(patched_image, patch_size):
    h,w = patched_image.shape
    x_coords = list(range(0,h,patch_size))
    y_coords = list(range(0,w,patch_size))
    nr = len(x_coords)
    nc = len(y_coords)
    depatched_image = np.zeros((nr,nc))
    for i in range(nr):
        for j in range(nc):
            depatched_image[i,j] = patched_image[x_coords[i], y_coords[j]]
    return depatched_image 


def repatchify(depatched_image, patch_size, original_size):
    h,w = original_size
    x_coords = list(range(0,h, patch_size))
    y_coords = list(range(0,w, patch_size))
    x_coords.append(h)
    y_coords.append(w)
    patched_image = np.zeros(original_size)
    for i in range(len(x_coords)-1):
        for j in range(len(y_coords)-1):
            patched_image[x_coords[i]:x_coords[i+1], y_coords[j]:y_coords[j+1]] = depatched_image[i,j]
    return patched_image


def get_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    method: str,
    agg_method: str,
    patch_size: int = 32,
    scale: int = 4,
    device: Union[str, torch.device] = "cpu",
    rgb_bands: Optional[List[int]] = [0, 1, 2],
    p_q = None
):
    """Estimate the distance between two tensors. All the distances
    are normalized to be between 0 and n,  where n is the maximum


    Args:
        x (torch.Tensor): The SR harmonized image (C, H, W).
        hr (torch.Tensor): The HR image (C, H, W).
        method (str): The method to use. Either "psnr" or "cpsnr".
        agg_method (str): The method to use to aggregate the distance.
            Either "pixel", "image", or "patch".
        patch_size (int, optional): The patch size to use if the patch
            method is used.
        scale (int, optional): The scale of the super-resolution.
        space_search (int, optional): This parameter is used to search
            for the best shift that maximizes the PSNR. By default, it is
            the same as the super-resolution scale.

    Returns:
        torch.Tensor: The metric value.
    """
    
    if x.shape[0] != y.shape[0]:
        raise ValueError("The number of channels in x and y must be the same.")

    if x.shape[1] != y.shape[1]:
        raise ValueError("The height of x and y must be the same.")
    if method == 'aggpatch':
 
        if p_q is None:
            raise ValueError('p and q not specified')
        p,q = p_q
        if q==1:
            distance_fn = L1(x=x, y=y, method='patch', patch_size=patch_size)
        elif q==2:
            distance_fn = L2(x=x, y=y, method='patch', patch_size=patch_size)
        else:
            raise ValueError('This q order is not supported for aggpatch')

        patch_distances = torch.tensor(depatchify(distance_fn.compute(), patch_size= patch_size)**(1/q))

        if p == np.inf:
            return torch.max(patch_distances)

        return torch.norm(patch_distances, p = p)

        

 
    if method == "l1":
        distance_fn = L1(x=x, y=y, method=agg_method, patch_size=patch_size)
    elif method == "l2":
        distance_fn = L2(x=x, y=y, method=agg_method, patch_size=patch_size)
    elif method == "struct":
        distance_fn = Struct(x=x, y=y, method=agg_method, patch_size=patch_size)
    elif method[:2]== "lp":
        p= int(method[2:])
        distance_fn = LP(x=x, y=y, p=p, method=agg_method, patch_size=patch_size)
   
    else:
        raise ValueError("No valid distance method.")

    return distance_fn.compute()



class ImgComparator:
    def __init__(self, fig, axlist = None):
        self.canvas = fig.canvas
        if axlist is None:
            self.axlist = fig.axes
        else:
            self.axlist = axlist
        self.cid_zoom = fig.canvas.mpl_connect('motion_notify_event', self.on_zoom)
    def on_zoom(self, event):
        if event.inaxes:
            xlim = event.inaxes.get_xlim()
            ylim = event.inaxes.get_ylim()
            for ax in self.axlist:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            self.canvas.draw_idle()



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


def build_S2_patched_dataset_DSHR(patchsize_X,img_dset_folder , subdatasets , out_dsfolder, labels = ('hr_data', 'lr_data'), border_X = 0, SR_factor = 4):
    border_Y = border_X//SR_factor
    patchsize_Y = patchsize_X//SR_factor

    hr_label = labels[0]
    lr_label = labels[1]


    for subds in subdatasets:
        print()
        print(f'Subdataset : {subds}')

        data_folder = os.path.join(img_dset_folder, subds)
        img_folders = os.listdir(data_folder)
        bar = [x for x in img_folders if 'json' not in x]

        for idxstr in tqdm(bar):
            img_folder = os.path.join(data_folder, idxstr)
            hr_path = f'{img_folder}/{hr_label}.tif'

            with rasterio.open(hr_path) as hr_src:
                hr_img = hr_src.read()
            hr_img = torch.from_numpy(hr_img) 
            
            lr_img = apply_upsampling(torch.tensor(hr_img), scale = SR_factor)

            #metrics.setup(lr=lr_img, sr=sr_img, hr=hr_img)

            patched_lr = get_patches_from_S2(lr_img, patchsize=patchsize_Y, border=border_Y)
            patched_hr = get_patches_from_S2(hr_img, patchsize_X, border_X)

            m = patched_lr.shape[0]
            for i in range(m):
                save_into_tiff(bands=np.array(patched_lr[i]), out_path=os.path.join(out_dsfolder, f'{subds}_{idxstr}_{i}_{lr_label}.tif'))
                save_into_tiff(bands=np.array(patched_hr[i]), out_path=os.path.join(out_dsfolder, f'{subds}_{idxstr}_{i}_{hr_label}.tif'))

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

