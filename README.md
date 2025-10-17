# Accuracy Bounds

This repository contains the official python-based implementation accompanying the paper:  
**“Computable Sharp Accuracy Bounds for Inverse Problems”**  ([arXiv:2510.10229](https://arxiv.org/abs/2510.10229))

If you use this software in your work, please cite our [preprint](https://arxiv.org/abs/2510.10229):

```bibtex
@article{gottschling2025average,
  title={Average Kernel Sizes--Computable Sharp Accuracy Bounds for Inverse Problems},
  author={Gottschling, Nina M and Iagaru, David and Gawlikowski, Jakob and Sgouralis, Ioannis},
  journal={arXiv preprint arXiv:2510.10229},
  year={2025}
}
```

For the definition of the worst-case kernel size, see [preprint](https://arxiv.org/abs/2311.16898):

```bibtex
@article{gottschling2023existence,
  title={On the existence of optimal multi-valued decoders and their accuracy bounds for undersampled inverse problems},
  author={Gottschling, Nina Maria and Campodonico, Paolo and Antun, Vegard and Hansen, Anders C},
  journal={arXiv preprint arXiv:2311.16898},
  year={2023}
}
```

## Installation

This project has been tested under **Python 3.7** on a Unix development environment.  

### 1. Clone the repository and create a virtual environment and change into the repository folder.
```
git clone https://github.com/nm19000/AccuracyBounds.git
cd AccuracyBounds
```

### 2. Create and activate a virtual environment
```
python -m venv venv
source venv/bin/activate
```
### 3. Install the project
#### 3.1 Lightweight Version (without PyTorch)
```
pip install -e .
```
#### 3.2 Full version (with PyTorch)
```
pip install -e .[torch]
```

# Average Kernel Sizes - Computable Sharp Accuracy Bounds for Inverse Problems

The reconstruction of an unknown quantity from noisy measurements is a mathematical problem relevant in most applied sciences, for example, in medical imaging, radar inverse scattering, or astronomy. This underlying mathematical problem is often an ill-posed (non-linear) reconstruction problem, referred to as an ill-posed inverse problem. To tackle such problems, there exist a myriad of methods to design approximate inverse maps, ranging from optimization-based approaches, such as compressed sensing, over Bayesian approaches, to data-driven techniques such as deep learning. For all stable approximate inverse maps, there are accuracy limits that are strictly larger than zero for ill-posed inverse problems, due to the accuracy-stability tradeoff [Gottschling et al., SIAM Review, 67.1 (2025)] and [Colbrook et al., Proceedings of the National Academy of Sciences, 119.12 (2022)]. The variety of methods that aim to solve such problems begs for a unifying approach to help scientists choose the approximate inverse map that obtains this theoretical optimum. Up to now there do not exist computable accuracy bounds to this optimum that are applicable to all inverse problems. We provide computable sharp accuracy bounds to the reconstruction error of solution methods to inverse problems. The bounds are method-independent and purely depend on the dataset of signals, the forward model of the inverse problem, and the noise model. To facilitate the use in scientific applications, we provide an algorithmic framework and an accompanying software library to compute these accuracy bounds.

## Accuracy Bounds for Inverse Problems

Computation of worst-case and average kernel size for an inverse problem with noise of the form: 

$$
\text{recover } x \in \mathcal{M}_1 \subset \mathbb{C}^{d_1} \text{ given noisy measurements } y = F(x,e)\in \mathbb{C}^{d_2} \text{ of } x  \text{ and }  e \in \\mathcal{E}\subset \mathbb{C}^{d_3}.
$$

The lower bound to the average error of any approximate inverse map is the average kernel size. The definition of and algorithms for computing the average kernel size are to be found in the following [preprint](https://arxiv.org/abs/2510.10229).

The lower bound to the worst-case error of any approximate inverse map is the worst-case kernel size. The definition of the worst-case kernel size can be found in the following [preprint](https://arxiv.org/abs/2311.16898).

# Testing 

The algorithms for computing the worst-case and average kernel size are tested against linear algebra examples, where the worst-case and average kernel size can be calculated analytically in the limit of infinite datapoints. Thus, we ensure that the implemented algorithms compute the correct quantities. The algorithm versions that run with cuda can be tested with ``test/test_toy_example_torch.py``. The algorithm versions that only run with numpy can be tested with ``test/test_toy_example_np.py``. All tests in the project can be automatically run by calling ``pytest``.

For computing the feasible sets there are two versions to compute these available: in terms of list and in terms of feasible appartenance matrices that allocate data points to feasible sets. Both versions are suitable for forward models with additive noise.

For interactive testing the algorithms to compute the worst-case and average kernel size with numpy and cuda please see ``test/playground/testing.ipynb``.

## Experiments from [Paper](https://arxiv.org/abs/2510.10229)

 We demonstrate the validity of the algorithms on two inverse problems from different domains: fluorescence localization microscopy and super-resolution of multi-spectral satellite data. The code for generating the data for reproducing the localization microscopy experiments can be found in ``examples/data/localization_microscopy/data_A/source_code``. The data used for the localization microscopy experiments can be found in examples/data/localization_microscopy. To reproduce the tabular data for the figures and tables in this manuscript, the average kernel size and loss computations, can be found in ``examples/example_localizationmicroscopy.ipynb``. The instructions for reproducing the satellite data super-resolution experiments are to be found below. 

### Satellite Data Super Resolution Experiments

Aditionally to this project, install the the [opensr-model library](https://github.com/ESAOpenSR/opensr-model) to perform the super-resolution inference with

pip install opensr-model

and the [opensr_test library](https://github.com/ESAOpenSR/opensr-test) to perform the tests with 

pip install opensr-test

The satellite data after the super-resolution inference should be stored under the following structure : 

cross_processed/ \
├── naip/ \
│ ├── 1/ \
│ │ ├── high resolution image (hr_res.tif) \
│ │ ├── low resolution image (lr_res, DSHR_res or lr .tif) \
│ │ ├── Prediction from high resolution image (sr_res or sr_res_fromDSHR .tif) \
│ ├── 2/ \
│ │ ├── ... \
│ ├── ... \
├── spain_crops/ \
│ ├── 1/ \
│ │ ├── ... \
│ ├── 2/ \
│ ├── ... \
├── spain_urban/ \
│ ├── 1/ \
│ │ ├── ... \
│ ├── 2/ \
│ ├── ... \
├── spot/ \
│ ├── 1/ \
│ │ ├── ... \
│ ├── 2/ \
│ │ ├── ... \
│ ├── ... \

The data used in our experiments can be downloaded from the indications on [huggingface](https://huggingface.co/datasets/isp-uv-es/opensr-test). 


### Preliminary Kernel Size Computations

Run the command
```
python test/S2_SR/Kernelsize_computations.py
```
with the desired options specified in the file ``Kernelsize_computations.py`` 

### Preliminary Operator Computations

To calculate the downsampling operator and its kernel projection in matrix form, run

```
python examples/S2_SR/op_testing.py
```

Manually change the values of the following variables inside the python file to enable or discard the corresponding computation or visualization : 

```
plot_sparsity = False # To plot the sparsity pattern of the operators \
check_DSOp = False  # To check that the downsampling operator uner matrix is correctly computed \
computeDS = False # To compute the downsampling operator under matrix form \
compute_P_null = False # To compute the kenrel projection \
check_P_null = False # To check that the kernel projection operator uner matrix is correctly computed\
scale_plot = False # To plot some satellite images with the scale bars \
```


### Run the experiments

After having run the preliminary kernel size and kernel projection operator computations, the experiments are ready to be reproduced with the command 

```
python test/S2_SR/experiments.py
```

Adjust the following parameters inside the python file: 
```
DSHR = True # Whether the lower resolution image is the downsampled version of the high resolution image (we run the experiments with DSHR = true)\
light_loading = False # Whether you use the light dataloader or you want to use stored patches. We run the experiments with the parameter set to false, but it is recommended for more memory and speed efficiency. Warning : it has to correspond with the value of the ``--light_load`` parameter in the kernel size computations . If light_loading is set to false, you will need to generate a dataset where each file corresponds to a patch. This can be done using the function ``build_S2_patched_dataset_DSHR`` or ``build_S2_patched_dataset`` in the ``utils_torch.py`` file. Warning such a patched dataset may contain more than 100 000 patches files for 119 full sized images. It is therefore not recommended. If activated, use the dataset SRDataset_perimg_lightload instead of the dataset SRDataset_perimg \
PS_X = 16  # Patch size in high resolution (has to correspond with the kernel size computations) \
PS_Y = PS_X//4 \
p_norm = 2 # Defines the used norm among the $L^p$ norms \
SR_factor = 4 # Leave it to 4 (super resolution factor) \
noise_level_KS = 4000 # has to Correspond with the preliminary computations of the kernel size \
preload_feas_info = True # Preload or not the feasible information from the feasible appartenance matrix. It has to be activated the first time so that the json file grouping the essential information from the feasible appartenance matrix can be saved (it may take more than 30 minutes for 100 000 patches in the dataset). \
pred_type = 'bicub' # The model used for the predictions ('diff' for the opensr-modelm 'bilin' for bilinear interpolation and 'bicub' for bicubic interpolation) \
p = 2 # The p parameter for kernel size and the loss computations
```

Modify the paths root_folder, feas_app_lightl_path etc manually in the file, according the needs.

The following functions can be activated or deactivated : 

- metrics_opensrtest checks the consistency of the predictions with the results shown by [opensr-test](https://github.com/ESAOpenSR/opensr-test)

- compute_LB_dists computes distances for the loss and the kernel size terms and stores them.

- get_LB_loss_points displays the half Kernesize lower bound and the Loss terms.

