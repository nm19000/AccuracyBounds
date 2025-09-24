import numpy as np
import random
import torch 
import os
from tqdm import tqdm
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


