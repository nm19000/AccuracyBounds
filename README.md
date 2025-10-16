# Accuracy Bounds

## Accuracy Bound for Inverse Problems

Install project with 

""
pip install -e .
""



# Accuracy Bounds

## Accuracy Bounds for Inverse Problems


Computation of worst-case and average kernel size from [1] for an inverse problem with noise of the form: 

$$
\text{recover } x \in \mathcal{M}_1 \subset \mathbb{C}^{d_1} \text{ given noisy measurements } y = F(x,e)\in \mathbb{C}^{d_2} \text{ of } x  \text{ and }  e \in \\mathcal{E}\subset \mathbb{C}^{d_3}.
\text{recover } x \in \mathcal{M}_1 \subset \mathbb{C}^{d_1} \text{ given noisy measurements } y = F(x,e)\in \mathbb{C}^{d_2} \text{ of } x  \text{ and }  e \in \\mathcal{E}\subset \mathbb{C}^{d_3}.
$$


### Algoritm for Allocating Signal Data into Feasible Sets


- Require  $K, N(K)_{\mathrm{max}} \in \mathbb{N}$, $\mathcal{M}_2$, $\mathcal{M}_1\times \mathcal{E}$, $F$
- $\mathcal{D} = \emptyset$
- For $k \in \{1,...,K\}$:
    - $y_k \in \mathcal{M}_2$, here samples $y_k$ can be either given as inputs or sampled during the algorithm.
    - $F_{y_k}^{N(k)} = \emptyset$, $N(k) = 0$
    - For $(x_{k,n},e_{k,n}) \in \mathcal{M}_1\times \mathcal{E}$ ( here the sampling strategy does not affect the validity of the accuracy bounds).
        - If $F(x_{k,n},e_{k,n})= y_k$ (Implementation of condition is forward model dependent.):
            - $F_{y_k}^{N} \gets F_{y_k}^{N} \bigcup x_{k,n}$
            - $\mathcal{D} \gets \mathcal{D} \bigcup (x_{k,n}, y_k)$
        - ElseIf $|F_{y_k}^{N}| \geq N(K)_{max}$:
            - Break 
        - EndIf 
    - EndFor
    - $N(k) = |F_{y_k}^{N}|$
    - $(N(k)) \gets \bigcup_k N(k)$
    - $(F_{y_k}^{N}) \gets \bigcup_k F_{y_k}^{N(k)}$
- EndFor 
- Return:  $(F_{y_k}^{N})$, $(N(k))$, $\mathcal{D}$

### Algoritm for Computing the Worst-Case Kernel Size From Feasible Sets

- Require $p \in (0,\infty)$, $K \in \mathbb{N}$, $(F_{y_k}^{N})$, $(N(k))$
- State $\text{Kersize}^w(F,\mathcal{M}_1,\mathcal{E},p)_K = 0$
- For $k \in \{1,...,K\}$:
    - $v_k = \emptyset$
    - If $N(k)\neq 0$:
        - While $x_{k,n},x_{k,n'} \in F_{y_k}$:
            - $v_k \gets v_k \bigcup \|x_{k,n} -x_{k,n'}\|^p$
        - EndWhile
            - $v_k = \max \|x_{k,n} -x_{k,n'}\|^p$
    - EndIf
- $\text{Kersize}^w(F,\mathcal{M}_1,\mathcal{E},p)_K = \max_k v_k$
- Return $\text{Kersize}^w(F,\mathcal{M}_1,\mathcal{E},p)_K$


### Algoritm for Computing the Average Kernel Size From Feasible Sets

- Require $p \in (0,\infty)$, $K \in \mathbb{N}$,  $(F_{y_k}^{N})$, $(N(k))$
- State $\text{Kersize}(F,\mathcal{M}_1,\mathcal{E},p)_K = 0$
- For $k \in \{1,...,K\}$:
    - If $N(k)\neq 0$:
        - While $x_{k,n},x_{k,n'} \in F_{y_k}$:
            - $v_k \gets v_k + \|x_{k,n} -x_{k,n'}\|^p$
        - EndWhile
            - $v_k = \frac{1}{N(k)^2}v_k$
    - ElseIf $N(k) = 0$:
        - $v_k = 0$
    - EndIf
    - $\text{Kersize}(F,\mathcal{M}_1,\mathcal{E},p)_K \gets \text{Kersize}(F,\mathcal{M}_1,\mathcal{E},p)_K+v_k$
- EndFor
- $\text{Kersize}(F,\mathcal{M}_1,\mathcal{E},p)_K=\left(\frac{1}{K} \text{Kersize}(F,\mathcal{M}_1,\mathcal{E},p)_K\right)^{1/p}$
- Return $\text{Kersize}(F,\mathcal{M}_1,\mathcal{E},p)_K$ 


### Algoritm for Allocating Signal Data into Feasible Sets


- Require  $K, N(K)_{\mathrm{max}} \in \mathbb{N}$, $\mathcal{M}_2$, $\mathcal{M}_1\times \mathcal{E}$, $F$
- $\mathcal{D} = \emptyset$
- For $k \in \{1,...,K\}$:
    - $y_k \in \mathcal{M}_2$, here samples $y_k$ can be either given as inputs or sampled during the algorithm.
    - $F_{y_k}^{N(k)} = \emptyset$, $N(k) = 0$
    - For $(x_{k,n},e_{k,n}) \in \mathcal{M}_1\times \mathcal{E}$ ( here the sampling strategy does not affect the validity of the accuracy bounds).
        - If $F(x_{k,n},e_{k,n})= y_k$ (Implementation of condition is forward model dependent.):
            - $F_{y_k}^{N} \gets F_{y_k}^{N} \bigcup x_{k,n}$
            - $\mathcal{D} \gets \mathcal{D} \bigcup (x_{k,n}, y_k)$
        - ElseIf $|F_{y_k}^{N}| \geq N(K)_{max}$:
            - Break 
        - EndIf 
    - EndFor
    - $N(k) = |F_{y_k}^{N}|$
    - $(N(k)) \gets \bigcup_k N(k)$
    - $(F_{y_k}^{N}) \gets \bigcup_k F_{y_k}^{N(k)}$
- EndFor 
- Return:  $(F_{y_k}^{N})$, $(N(k))$, $\mathcal{D}$

### Algoritm for Computing the Worst-Case Kernel Size From Feasible Sets

- Require $p \in (0,\infty)$, $K \in \mathbb{N}$, $(F_{y_k}^{N})$, $(N(k))$
- State $\text{Kersize}^w(F,\mathcal{M}_1,\mathcal{E},p)_K = 0$
- For $k \in \{1,...,K\}$:
    - $v_k = \emptyset$
    - If $N(k)\neq 0$:
        - While $x_{k,n},x_{k,n'} \in F_{y_k}$:
            - $v_k \gets v_k \bigcup \|x_{k,n} -x_{k,n'}\|^p$
        - EndWhile
            - $v_k = \max \|x_{k,n} -x_{k,n'}\|^p$
    - EndIf
- $\text{Kersize}^w(F,\mathcal{M}_1,\mathcal{E},p)_K = \max_k v_k$
- Return $\text{Kersize}^w(F,\mathcal{M}_1,\mathcal{E},p)_K$


### Algoritm for Computing the Average Kernel Size From Feasible Sets

- Require $p \in (0,\infty)$, $K \in \mathbb{N}$,  $(F_{y_k}^{N})$, $(N(k))$
- State $\text{Kersize}(F,\mathcal{M}_1,\mathcal{E},p)_K = 0$
- For $k \in \{1,...,K\}$:
    - If $N(k)\neq 0$:
        - While $x_{k,n},x_{k,n'} \in F_{y_k}$:
            - $v_k \gets v_k + \|x_{k,n} -x_{k,n'}\|^p$
        - EndWhile
            - $v_k = \frac{1}{N(k)^2}v_k$
    - ElseIf $N(k) = 0$:
        - $v_k = 0$
    - EndIf
    - $\text{Kersize}(F,\mathcal{M}_1,\mathcal{E},p)_K \gets \text{Kersize}(F,\mathcal{M}_1,\mathcal{E},p)_K+v_k$
- EndFor
- $\text{Kersize}(F,\mathcal{M}_1,\mathcal{E},p)_K=\left(\frac{1}{K} \text{Kersize}(F,\mathcal{M}_1,\mathcal{E},p)_K\right)^{1/p}$
- Return $\text{Kersize}(F,\mathcal{M}_1,\mathcal{E},p)_K$ 

# References
If you use this software in your work, please cite our [paper](https://arxiv.org/abs/2510.10229)

```bibtex
@article{gottschling2025average,
  title={Average Kernel Sizes--Computable Sharp Accuracy Bounds for Inverse Problems},
  author={Gottschling, Nina M and Iagaru, David and Gawlikowski, Jakob and Sgouralis, Ioannis},
  journal={arXiv preprint arXiv:2510.10229},
  year={2025}
}
```

# Experiments from [paper](https://arxiv.org/abs/2510.10229)

### Satellite Data Super Resolution Experiments

Aditionally install the the opensr-model library to perform the Super Resolution inference https://github.com/ESAOpenSR/opensr-model

pip install opensr-model

and the opensr_test to perform the tests : https://github.com/ESAOpenSR/opensr-test 

pip install opensr-test


It is assumed that the Satellite data after Super resolution inference is stored under the following structure : 

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

The data used in our experiments can be downloaded from the indications on https://huggingface.co/datasets/isp-uv-es/opensr-test 


The folders and the files cqn be renamed according to the code

## Preliminary Kernel size computations

Run the command
 " python test/S2_SR/Kernelsize_computations.py "
 with the desired options specified in the file Kernelsize_computations.py 

## Preliminary operator calculation

To calculate the Downsampling operator and its kernel projection under matrix form, run

" python examples/S2_SR/op_testing.py "

Change manually the values of the following variables inside the python file to enable or discard the corresponding computation or visualization : 

plot_sparsity = False # To plot the sparsity pattern of the operators \
check_DSOp = False  # To check that the downsampling operator uner matrix is correctly computed \
computeDS = False # To compute the downsampling operator under matrix form \
compute_P_null = False # To compute the Null space projection \
check_P_null = False # To check that the Null space projection operator uner matrix is correctly computed\
scale_plot = False # To plot some satellite images with the scale bars \

The path of the dataset can as wall be changed according to the needs.

## Run the experiments

After having run the preliminary Kernel size and Kernel projection operator computations, the experiments are ready to be reproduced with the command 

""
python test/S2_SR/experiments.py
""

Adjust inside the python file the following parameters : 

DSHR = True # Whether the lower resolution image is the downsampled version of the high resolution image (we run the experiments with DSHR = true)\
light_loading = False # Whether you use the light dataloader or you want to use stored patches. We run the experiments with the parameter set to false, but it is recommended for more memory and speed efficiency. Warning : it has to correspond with the value of the --light_load parameter in the kernel size computations . If light_loading is set to false, you will need to generate a dataset where each file corresponds to a patch. This can be done using the function build_S2_patched_dataset_DSHR or build_S2_patched_dataset in the utils.py file. Warning such a patched dataset may contain more than 100 000 patches files for 119 full sized images. It is therefore not recommended. If activated, use the dataset SRDataset_perimg_lightload instead of the dataset SRDataset_perimg \
PS_X = 16  # Patch size in high resolution (has to correspond with the kernel size computations) \
PS_Y = PS_X//4 \
p_norm = 2 # Defines the used norm among the $L^p$ norms \
SR_factor = 4 # Leave it to 4 (super resolution factor) \
noise_level_KS = 4000 # has to Correspond with the preliminary computations of the kernel size \
preload_feas_info = True # Preload or not the feasible information from the feasible appartenance matrix. It has to be activated the first time so that the json file grouping the essential information from the feasible appartenance matrix can be saved (it may take more than 30 minutes for 100 000 patches in the dataset). \
pred_type = 'bicub' # The model used for the predictions ('diff' for the opensr-modelm 'bilin' for bilinear interpolation and 'bicub' for bicubic interpolation) \
p = 2 # The p parameter for Kernelsize and the loss computations


Modify the paths root_folder, feas_app_lightl_path etc manually in the file, according the needs.

The following functions can be activated or deactivated : 

- metrics_opensrtest checks the consistency of the predictions with the results shown by opensr-test in https://github.com/ESAOpenSR/opensr-test

- compute_LB_dists computes distances for the loss and the Kernelsize terms and stores them.

- get_LB_loss_points displays the half Kernesize lower bound and the Loss terms

