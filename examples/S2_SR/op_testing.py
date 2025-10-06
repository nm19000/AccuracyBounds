from accuracy_bounds.inverseproblems.utils import MatrixOpCalculator
from utils import DS_operator_32
from scipy import sparse
from scipy.sparse import hstack, identity, load_npz
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import torch
import rasterio
from utils import apply_square_op_small, apply_square_op_full, rescale_plot, ImgComparator, get_distance

from opensr_test.config import Config
from opensr_test.main import Metrics

if __name__ == '__main__':
    plot_sparsity = False # To plot the sparsity pattern of the operators
    check_DSOp = False  # To check that the downsampling operator uner matrix is correctly computed
    computeDS = False # To compute the downsampling operator under matrix form
    compute_P_null = False # To compute the Null space projection 
    check_P_null = False # To check that the Null space projection operator uner matrix is correctly computed
    scale_plot = False # To plot some satellite images with the scal bars

    border = 16

    OP_calculator = MatrixOpCalculator(n_in=128*128, n_out = 32*32, Operator=DS_operator_32)


    if computeDS:
        A_sparse = OP_calculator.build_sparse_matrix_parallel()
        sparse.save_npz('DS_op_32.npz', A_sparse)
    else:
        A_sparse = load_npz('../Operators/DS_op_32.npz')

    if compute_P_null:
    
        # Compute null space basis (sparse-aware)
        range_basis = OP_calculator.get_range_space_basis(A_sparse, sigma_threshold_ratio=0.0001)

        # Build projection operator onto null space (as a LinearOperator, which is memory efficient)
        P_null = OP_calculator.make_null_projection_operator(range_basis)

        np.save('P_null_32.npy', P_null)
    else:
        P_null = np.load('../Operators/P_null_32.npy')

    w,h = P_null.shape


    if plot_sparsity:
        plt.spy(A_sparse, markersize=1)
        plt.title("Sparsity pattern of A")
        plt.show()

        


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

            hr_null = apply_square_op_full(P_null, metrics.hr, out_2D_shape_op=(128,128), border = border)
            hr_bis = metrics.hr-2*hr_null

            if check_P_null:
                p_X = 1
                left_side = torch.norm(hr_null, p = 1)**2
                right_side = (torch.norm(metrics.hr-metrics.sr, p = 1)**2 + torch.norm(hr_bis-metrics.sr, p = 1)**2)*0.5
                print(f'{left_side} <= {right_side} ')
                print(f'normalized : {left_side/(left_side+right_side)} <= {right_side/(left_side+right_side)}')
                print(left_side <= right_side)

                A__pnull_x = torch.tensor(metrics.apply_upsampling(hr_null, metrics.scale_factor))

                norm_A_pnull_x = get_distance(A__pnull_x, torch.zeros_like(A__pnull_x), method='l2', agg_method='patch', patch_size=12//4, scale = metrics.scale_factor, device = metrics.params.device)

                norm_Ax = get_distance(metrics.lr, torch.zeros_like(metrics.lr), method='l2', agg_method='patch', patch_size=12//4, scale = metrics.scale_factor, device = metrics.params.device)
                

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

            if scale_plot:
            
                GSD_LR = 10
                GSD_HR = 2.5
                scale_bar_real_length = 200


                fig, ax = plt.subplots(figsize=(10, 10))
                c,h,w = metrics.lr.shape
                ax.imshow(metrics.lr.permute(1,2,0)/3000)
                print('Left : LR image')
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Scale bar properties
                scale_bar_length = 20  # Length of the scale bar in pixels
                scale_bar_label = "200 m"  # Label for the scale bar
                bar_color = "black"  # Color of the scale bar
                bar_thickness = 3  # Thickness of the scale bar

                # Position: bottom right corner
                height, width = metrics.lr.shape[1:]
                x_start = width - scale_bar_length - 5  # 20 pixels from the right
                x_end = width - 5  # 20 pixels from the right
                y_pos = height - 8  # 30 pixels from the bottom

                # Draw the scale bar (a line with ticks)
                plt.plot([x_start, x_end], [y_pos, y_pos], color=bar_color, linewidth=bar_thickness)

                # Add ticks to the scale bar
                tick_spacing = 20  # Space between ticks in pixels
                num_ticks = (x_end - x_start) // tick_spacing  # Calculate how many ticks

                for i in range(num_ticks + 1):
                    # Each tick is positioned along the scale bar at equal intervals
                    tick_x = x_start + i * tick_spacing
                    plt.plot([tick_x, tick_x], [y_pos - 2, y_pos + 2], color=bar_color, linewidth=3)  # Draw tick

                # Add the label to the scale bar
                plt.text((x_start + x_end) / 2, y_pos - 3, scale_bar_label, color=bar_color, ha='center', va='bottom', fontsize=12, fontweight='bold')

                plt.show()




                c,h,w = hr_bis.shape
                plt.imshow(metrics.hr.permute(1,2,0)/3000)

                print('Middle : HR image')
                plt.xticks([])
                plt.yticks([])

                # Scale bar properties
                scale_bar_length = 80  # Length of the scale bar in pixels
                scale_bar_label = "200 m"  # Label for the scale bar
                bar_color = "black"  # Color of the scale bar
                bar_thickness = 3  # Thickness of the scale bar

                # Position: bottom right corner
                height, width = hr_bis.shape[1:]
                x_start = width - scale_bar_length - 20  # 20 pixels from the right
                x_end = width - 20  # 20 pixels from the right
                y_pos = height - 30  # 30 pixels from the bottom

                # Draw the scale bar (a line with ticks)
                plt.plot([x_start, x_end], [y_pos, y_pos], color=bar_color, linewidth=bar_thickness)

                # Add ticks to the scale bar
                tick_spacing = 80  # Space between ticks in pixels
                num_ticks = (x_end - x_start) // tick_spacing  # Calculate how many ticks

                for i in range(num_ticks + 1):
                    # Each tick is positioned along the scale bar at equal intervals
                    tick_x = x_start + i * tick_spacing
                    plt.plot([tick_x, tick_x], [y_pos - 5, y_pos + 5], color=bar_color, linewidth=3)  # Draw tick

                # Add the label to the scale bar
                plt.text((x_start + x_end) / 2, y_pos - 10, scale_bar_label, color=bar_color, ha='center', va='bottom', fontsize=12, fontweight='bold')
                plt.show()


                plt.imshow(hr_bis.permute(1,2,0)/3000)
                print('Right : HR sym image')
                plt.xticks([])
                plt.yticks([])

                # Scale bar properties
                scale_bar_length = 80  # Length of the scale bar in pixels
                scale_bar_label = "200 m"  # Label for the scale bar
                bar_color = "black"  # Color of the scale bar
                bar_thickness = 3  # Thickness of the scale bar

                # Position: bottom right corner
                height, width = hr_bis.shape[1:]
                x_start = width - scale_bar_length - 20  # 20 pixels from the right
                x_end = width - 20  # 20 pixels from the right
                y_pos = height - 30  # 30 pixels from the bottom

                # Draw the scale bar (a line with ticks)
                plt.plot([x_start, x_end], [y_pos, y_pos], color=bar_color, linewidth=bar_thickness)

                # Add ticks to the scale bar
                tick_spacing = 80  # Space between ticks in pixels
                num_ticks = (x_end - x_start) // tick_spacing  # Calculate how many ticks

                for i in range(num_ticks + 1):
                    # Each tick is positioned along the scale bar at equal intervals
                    tick_x = x_start + i * tick_spacing
                    plt.plot([tick_x, tick_x], [y_pos - 5, y_pos + 5], color=bar_color, linewidth=3)  # Draw tick

                # Add the label to the scale bar
                plt.text((x_start + x_end) / 2, y_pos - 10, scale_bar_label, color=bar_color, ha='center', va='bottom', fontsize=12, fontweight='bold')

                plt.show()
            

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
