import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import hstack, identity, load_npz
import rasterio
from tqdm import tqdm

import sys
sys.path.append('../../../opensr-test')
#from utils import MatrixOpCalculator

sys.path.append('../../../opensr-test/opensr_test')
sys.path.append('../../src/inverseproblems')

from opensr_test.config import Config
from opensr_test.main import Metrics

from pdb import set_trace
import json
import os
from typing import List, Optional, Union
from abc import ABC, abstractmethod


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



def DS_operator_full(HR_flat):

    HR = torch.tensor(HR_flat.reshape((1, 1, 512, 512)))  # Shape: (batch=1, channels=1, H, W)
    LR = torch.nn.functional.interpolate(
        input=HR, scale_factor=1 / 4, mode="bilinear", antialias=True
    ).squeeze(0).squeeze(0)  # Remove batch and channel dims, shape: (128, 128)
    return np.asarray(LR).flatten()

def DS_operator_32(HR_flat):
    HR = torch.tensor(HR_flat.reshape(1,1,128,128))
    LR = torch.nn.functional.interpolate(
        input=HR, scale_factor=1 / 4, mode="bilinear", antialias=True
    ).squeeze(0).squeeze(0)
    return np.asarray(LR).flatten()

def padding(img, target_shape):
    pad_y = (target_shape[1] - img.shape[1])  # 112
    pad_x = (target_shape[2] - img.shape[2])  # 112

    # Divide padding equally (top/bottom, left/right)
    pad_top = pad_y // 2       # 56
    pad_bottom = pad_y - pad_top  # 56

    pad_left = pad_x // 2      # 56
    pad_right = pad_x - pad_left  # 56

    # Apply padding
    img_padded = np.pad(
        img,
        pad_width=((0,0),(pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',  # or 'edge', 'reflect', etc.
        constant_values=0
    )
    return img_padded

def apply_rect_op_small(Op_Mat, imgX, img_e, out_2Dshape):
    '''
    Same as apply_square_op_small but here, the operator is non  square and has 2 inputs : x and noise e.
    Both x and e have C channels and Op_mat.shape[1] = sum of lengts of vectoes associates with x and e
    '''
    matlist = []
    for i in range(imgX.shape[0]):
        imgX_flat = np.asarray(imgX[i]).flatten()
        img_e_flat = np.asarray(img_e[i]).flatten()
        input_vec = np.concatenate([imgX_flat, img_e_flat])
        matlist.append((Op_Mat@input_vec).reshape(out_2Dshape))
    return torch.tensor(np.stack(matlist))

def apply_square_op_small(Op_Mat,img, out_2Dshape):
    '''
    The image has a shape of C,h,w
    Function to apply an opertor into a 2d mateix form, to an image in 2d * channels. 
    For each channel, it flattens the the channel into a 1d vector, before applying matrix multiplication and reshaping to the desired 2d shape. Note that reshape is the inverse function of flatten.
    '''
    matlist = []
    for i in range(img.shape[0]):
        matlist.append((Op_Mat@np.asarray(img[i]).flatten()).reshape(out_2Dshape))
    return torch.tensor(np.stack(matlist))

def apply_rect_op_full(Op_mat, imgX, img_e, out_2D_shape_op, border = 4, scale=4):
    '''
    Here, the operator is rectangular, but the output is still the same shape than the image. It applies to pi_1(P_(N(F))) for example
    '''
    c,h,w = imgX.shape
    b,a = out_2D_shape_op
    n_y = (h)//(b-2*border)
    n_x = (w)//(a-2*border)

    OP_img = torch.zeros_like(imgX)

    for i in range(n_y):
        for j in range(n_x):
            # idx, operator, place
            imin = i*(b-2*border)
            imax = imin + b

            jmin = j*(a-2*border)
            jmax = jmin + a
            if jmax<=w and imax<=h:
                Apatch = apply_rect_op_small(Op_mat, imgX[:, imin:imax, jmin:jmax], img_e[:, imin//scale: imax//scale, jmin//scale:jmax//scale], out_2Dshape=out_2D_shape_op)
                OP_img[:,imin+border:imax-border,jmin+border: jmax-border ] = Apatch[:,border: -border, border:-border]
    # Do it for the last row, col
    for i in range(n_y):
        imin = i*(b-2*border)
        imax = imin + b

        jmin = h-a
        jmax = h

        if imax <= h:
            Apatch = apply_rect_op_small(Op_mat, imgX[:,imin:imax, jmin:jmax], img_e[:, imin//scale: imax//scale, jmin//scale:jmax//scale], out_2Dshape=out_2D_shape_op)
            OP_img[:,imin+border:imax-border,jmin+border: jmax ] = Apatch[:,border: -border, border:]
    
    for j in range(n_x):
        jmin = j*(a-2*border)
        jmax = jmin + a

        imin = w-b
        imax = w

        if jmax <= w:
            Apatch = apply_rect_op_small(Op_mat, imgX[:,imin:imax, jmin:jmax], img_e[:, imin//scale: imax//scale, jmin//scale:jmax//scale], out_2Dshape=out_2D_shape_op)
            OP_img[:,imin+border:imax,jmin+border: jmax-border ] = Apatch[:,border: , border:-border]
    
    Apatch = apply_rect_op_small(Op_mat, imgX[:,w-b:w, h-a:h],img_e[:, imin//scale: imax//scale, jmin//scale:jmax//scale], out_2Dshape=out_2D_shape_op)
    OP_img[:,w-b+border:w,h-a+border: h ] = Apatch[:,border: , border:]
    return OP_img


def apply_square_op_full(Op_mat, img, out_2D_shape_op, border = 4):
    '''
    The operator has to be square. It applies to P_{N(A)} for example
    '''
    c,h,w = img.shape
    b,a = out_2D_shape_op
    n_y = (h)//(b-2*border)
    n_x = (w)//(a-2*border)

    OP_img = torch.zeros_like(img)
    for i in range(n_y):
        for j in range(n_x):
            # idx, operator, place
            imin = i*(b-2*border)
            imax = imin + b

            jmin = j*(a-2*border)
            jmax = jmin + a

            if jmax<=w and imax<=h:
                if i==0 and j ==0:
                    Apatch = apply_square_op_small(Op_mat, img[:,imin:imax, jmin:jmax], out_2Dshape=out_2D_shape_op)
                    OP_img[:,imin:imax-border,jmin: jmax-border ] = Apatch[:,: -border, :-border]
                elif i==0 and j >0:
                    Apatch = apply_square_op_small(Op_mat, img[:,imin:imax, jmin:jmax], out_2Dshape=out_2D_shape_op)
                    OP_img[:,imin:imax-border,jmin+border: jmax-border ] = Apatch[:,: -border, border:-border]
                elif i>0 and j==0:
                    Apatch = apply_square_op_small(Op_mat, img[:,imin:imax, jmin:jmax], out_2Dshape=out_2D_shape_op)
                    OP_img[:,imin+border:imax-border,jmin: jmax-border ] = Apatch[:,border: -border, :-border]
                
                else :
                    Apatch = apply_square_op_small(Op_mat, img[:,imin:imax, jmin:jmax], out_2Dshape=out_2D_shape_op)
                    OP_img[:,imin+border:imax-border,jmin+border: jmax-border ] = Apatch[:,border: -border, border:-border]

    # Do it for the last row, col
    for i in range(n_y):
        imin = i*(b-2*border)
        imax = imin + b

        jmin = h-a
        jmax = h

        if imax <= h:
            if i==0:
                Apatch = apply_square_op_small(Op_mat, img[:,imin:imax, jmin:jmax], out_2Dshape=out_2D_shape_op)
                OP_img[:,imin:imax-border,jmin+border: jmax ] = Apatch[:,: -border, border:]
            else:
                Apatch = apply_square_op_small(Op_mat, img[:,imin:imax, jmin:jmax], out_2Dshape=out_2D_shape_op)
                OP_img[:,imin+border:imax-border,jmin+border: jmax ] = Apatch[:,border: -border, border:]
    
    for j in range(n_x):
        jmin = j*(a-2*border)
        jmax = jmin + a

        imin = w-b
        imax = w

        if jmax <= w:
            if j==0:
                Apatch = apply_square_op_small(Op_mat, img[:,imin:imax, jmin:jmax], out_2Dshape=out_2D_shape_op)
                OP_img[:,imin+border:imax,jmin: jmax-border ] = Apatch[:,border: , :-border]
            else:
                Apatch = apply_square_op_small(Op_mat, img[:,imin:imax, jmin:jmax], out_2Dshape=out_2D_shape_op)
                OP_img[:,imin+border:imax,jmin+border: jmax-border ] = Apatch[:,border: , border:-border]
    
    Apatch = apply_square_op_small(Op_mat, img[:,w-b:w, h-a:h], out_2Dshape=out_2D_shape_op)
    OP_img[:,w-b+border:w,h-a+border: h ] = Apatch[:,border: , border:]
    return OP_img


def rescale_plot(img):
    minval = torch.min(img)
    maxval = torch.max(img)
    return (img-minval)/(maxval-minval)


def draw_img_noise(image,alpha_poiss,sigma_gauss = None, type = 'poisson', bands_OI = [3,2,1,7]):

    shp = image.shape
    if type == 'gauss':
        noise_raw = torch.randn(shp)
        noise_OI = noise_raw[bands_OI] * sigma_gauss[:, None, None]
        noise = noise_raw
        noise[bands_OI] = noise_OI

        image_tensor = torch.tensor(image, dtype=torch.float32)
    
        return torch.tensor(image_tensor + noise)


    elif type == 'poisson':
        alpha = alpha_poiss
        image_tensor = torch.tensor(image, dtype=torch.float32)
        return torch.poisson(image_tensor/alpha) *alpha






if __name__ == '__main__':
    plot_sparsity = False
    check_DSOp = False
    computeDS = False
    compute_P_null = False
    check_P_null = False
    check_P_null_concat = False

    border = 16

    #OP_calculator = MatrixOpCalculator(n_in=128*128, n_out = 32*32, Operator=DS_operator_32)


    if computeDS:
        A_sparse = OP_calculator.build_sparse_matrix_parallel()
        sparse.save_npz('DS_op_32.npz', A_sparse)
    else:
        A_sparse = load_npz('../Operators/DS_op_32.npz')
        F_sparse = hstack([A_sparse, identity(A_sparse.shape[0], format='csc')], format='csc')
        #sparse.save_npz('F_op_32.npz', F_sparse)

        h,w = A_sparse.shape

    if compute_P_null:
    
        if True:
            # Compute null space basis (sparse-aware)
            range_basis = OP_calculator.get_range_space_basis(A_sparse, sigma_threshold_ratio=0.0001)

            # Build projection operator onto null space (as a LinearOperator, which is memory efficient)
            P_null = OP_calculator.make_null_projection_operator(range_basis)

            np.save('P_null_32.npy', P_null)
        else:
            # Compute null space basis (sparse-aware)
            range_basis = OP_calculator.get_range_space_basis(F_sparse)

            # Build projection operator onto null space (as a LinearOperator, which is memory efficient)
            P_null = OP_calculator.make_null_projection_operator(range_basis)

            np.save('F_null_32.npy', P_null)

    P_null = np.load('../Operators/P_null_32.npy')
    F_null = np.load('../Operators/F_null_32.npy')

    w,h = P_null.shape
    F_null_restricted = F_null[:w, :h]


    if plot_sparsity:
        if computeDS or compute_P_null:
            plt.spy(A_sparse, markersize=1)
            plt.title("Sparsity pattern of A")
            plt.show()
        plt.imshow(F_null[:1000, :1000])
        plt.title("F_null viz")
        plt.show()
        set_trace()
        plt.imshow(F_sparse[:,-2000:])
        plt.title('F_sparse')
        plt.show()
        #plt.spy(F_sparse)
        #plt.title("Sparsity pattern of F")
        #plt.show()
        


    dset_folder = '../sat_data/cross_processed'
    subdsets = ['spain_crops','naip' ,'spain_urban','spot']
    

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

            if True:
                hr_null = apply_square_op_full(P_null, metrics.hr, out_2D_shape_op=(128,128), border = border)
                hr_bis = metrics.hr-2*hr_null

                p_X = 1
                left_side = torch.norm(hr_null, p = 1)**2
                right_side = (torch.norm(metrics.hr-metrics.sr, p = 1)**2 + torch.norm(hr_bis-metrics.sr, p = 1)**2)*0.5
                print(f'{left_side} <= {right_side} ')
                print(f'normalized : {left_side/(left_side+right_side)} <= {right_side/(left_side+right_side)}')
                print(left_side <= right_side)

                A__pnull_x = torch.tensor(metrics.apply_upsampling(hr_null, metrics.scale_factor))

                norm_A_pnull_x = get_distance(A__pnull_x, torch.zeros_like(A__pnull_x), method='l2', agg_method='patch', patch_size=12//4, scale = metrics.scale_factor, device = metrics.params.device)

                norm_Ax = get_distance(metrics.lr, torch.zeros_like(metrics.lr), method='l2', agg_method='patch', patch_size=12//4, scale = metrics.scale_factor, device = metrics.params.device)
                #plt.imshow(norm_Ax)
                #plt.colorbar()
                #plt.show()

                fig, axes = plt.subplots(1,2)
                axes[0].imshow(A__pnull_x.permute(1,2,0))
                axes[0].set_title(r' $A.P_{null}(x)$')

                #axes[1].set_title(r'$\pi_1(P_{N(F)}^{\perp}-P_{N(F)})(x,0)$')
                axes[1].set_title(r' $\|A.P_{null}(x)\|$ per patch')

                im = axes[1].imshow(norm_A_pnull_x)
                im.set_clim(0, 40)
                fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

                print(f'cosine  (x_null, x-x_null) : {torch.dot(hr_null.flatten(), (metrics.hr - hr_null).flatten()) / (torch.norm(hr_null.flatten(),2) * torch.norm((metrics.hr - hr_null).flatten(),2))}')
                plt.show()

            else:
                lr_img_noisy = draw_img_noise(lr_img, opt_alpha, type = 'poisson' )
                noise_img = lr_img_noisy-lr_img

                hr_Fnull = apply_rect_op_full(F_null[:w,:], metrics.hr,noise_img, out_2D_shape_op=(128,128), border = border)
                hr_bis = metrics.hr-2*hr_Fnull

            fig, axes = plt.subplots(1,2)
            axes[0].imshow(metrics.hr.permute(1,2,0)/3000)
            axes[0].set_title(r'HR img ($x$)')

            axes[1].imshow((hr_bis).permute(1,2,0)/3000)
            axes[1].set_title(r'$\pi_1(P_{N(F)}^{\perp}-P_{N(F)})(x,0)$')

            comparator = ImgComparator(fig)
            plt.show()
            #set_trace()

            if check_DSOp:
                hr_test=  torch.tensor(metrics.hr[:, 100:228, 100:228])

                #Testing whether we computed well the forward operator under matrix form
                DS_img_original = metrics.apply_upsampling(torch.tensor(hr_test),metrics.scale_factor).squeeze(0)

                DS_img_mat = apply_square_op_small(A_sparse, hr_test, out_2Dshape=DS_img_original.shape[1:])

                fig, axes = plt.subplots(2,2)
                axes[0,0].imshow(hr_test.permute(1,2,0)/3000)
                axes[0,0].set_title('HR img')

                axes[0,1].imshow(DS_img_original.permute(1,2,0)/3000)
                axes[0,1].set_title('DS img with original operator')

                axes[1,0].imshow(DS_img_mat.permute(1,2,0)/3000)
                axes[1,0].set_title('DS img with matrix operator')

                axes[1,1].imshow(np.abs(DS_img_mat-DS_img_original).permute(1,2,0)/3000)
                axes[1,1].set_title('Delta')

                comparator = ImgComparator(fig, axlist = [axes[0,1], axes[1,0], axes[1,1]])

                plt.show()

            if check_P_null:
                hr_padded =  torch.tensor(metrics.hr[:, 100:228, 100:228])
                lr_padded = apply_square_op_small(A_sparse, hr_padded, out_2Dshape=(32,32))
                hr_null = apply_square_op_small(P_null,hr_padded, out_2Dshape=(128,128))
            
                A_hrnull = apply_square_op_small(A_sparse, hr_null, (32,32))

                
                fig, axes = plt.subplots(2,2)
                axes[0,0].imshow(rescale_plot(hr_null.permute(1,2,0)))
                axes[0,0].set_title(f'P null(HR) \n Range 1-99 percent = {torch.quantile(hr_null, 0.01):2f}- {torch.quantile(hr_null, 0.99):2f}')

                axes[0,1].imshow(rescale_plot(A_hrnull.permute(1,2,0)))
                axes[0,1].set_title(f'A.P_null(HR)  \n Range 1-99 percent = {torch.quantile(A_hrnull, 0.01):2f}- {torch.quantile(A_hrnull, 0.99):2f}')

                axes[1,0].imshow(rescale_plot(lr_padded).permute(1,2,0))
                axes[1,0].set_title(f'A.HR')

                axes[1,1].imshow(rescale_plot(lr_padded-A_hrnull ).permute(1,2,0))
                axes[1,1].set_title(f'A.(HR-P_null(HR))')

                plt.show()

            if check_P_null_concat : 
                hr_null = apply_square_op_small(P_null, metrics.hr, out_2D_shape_op=(128,128), border = border)
                AHR_null = metrics.apply_upsampling(hr_null,metrics.scale_factor).squeeze(0)

                fig, axes = plt.subplots(1,2)
                axes[0].imshow(rescale_plot(hr_null.permute(1,2,0))[border:-border, border:-border,:])
                axes[0].set_title('P_null(HR) concatenated')

                axes[1].imshow(rescale_plot(AHR_null.permute(1,2,0))[border//4+1:-border//4-1, border//4+1:-border//4-1,:])
                axes[1].set_title('A.P_null(HR)')
                print(torch.max(AHR_null[:,border//4+1:-border//4-1, border//4+1:-border//4-1]))

                plt.show()

