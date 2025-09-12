# Accuracy Bounds

## Accuracy Bounds for Inverse Problems


Computation of worst-case and average kernel size from [1] for an inverse problem with noise of the form: 

$$
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
            - $F_{y_k}^{N(k)} \gets F_{y_k}^{N(k)} \bigcup \{x_{k,n}\}$
            - $\mathcal{D} \gets \mathcal{D} \bigcup \{(x_{k,n}, y_k)\}$
        - ElsIf $|F_{y_k}^{N(k)}| \geq N(K)_{max}$:
            - Break 
        - EndIf 
    - EndFor
    - $N(k) = |F_{y_k}^{N(k)}|$
    - $\{N(k)\}_{k=1}^K \gets \bigcup_k N(k)$
    - $\{F_{y_k}^{N(k)}\}_{k=1}^K \gets \bigcup_k F_{y_k}^{N(k)}$
- EndFor 
- Return:  $\{F_{y_k}^{N(k)}\}_{k=1}^K$, $\{N(k)\}_{k=1}^K$, $\mathcal{D}$



### Algoritm for computing the average kernel size

1) Randomly with any distribution $\mu$ sample 
$$
\{y_i\}_{i=1}^k \subset \mathcal{M}_2 = A(\mathcal{M}_1)+\mathcal{E}
$$
for $k \in \mathbb{N}$. For increasing $k$ only new $y_i \in \mathcal{M}_2$ are added and the old sampled points are kept.
2) For each $i \in \{1,...,k\}$, set $F^0_{y_i} = \emptyset$. Then, iteratively for fixed $n=n(k) \in \mathbb{N}$ for randomly sampled $x_n \in \mathcal{M}_1$, if
        
$$
e_n:= F(x_n,0)- y_i \in \mathcal{E},
$$

let $F^n_{y_i}  = F^{n-1}_{y_i}  \cup \{x_n\}$ with the $x_n$ choosen as above. Set 
$$
    ||v_n^i||= ||\pi_1((I-F^{t}F)(x_n,e_n))||.
$$

3) Now obtain the approximate average case kernel size by 
$$
\text{ker}^a(F, \mathcal{M}_1, \mathcal{E},p)_{k}^p = (F_*\mu)(\mathcal{M}_2) 2^p/k \sum_{i=1}^k 1/n(i) \sum_{l=1}^{n(i)} || v_l^i ||^p.
$$
4) For any $\delta >0$, obtain confidence intervals $c(k,\delta) = c(k,\delta,F, \mathcal{M}_1, \mathcal{E},p)$ with probability at least $1-c(k,\delta)$ using Hoeffding's inequality for exchangeable random variables.

# References
[1] Gottschling, Nina Maria, et al. "On the existence of optimal multi-valued decoders and their accuracy bounds for undersampled inverse problems." arXiv preprint arXiv:2311.16898 (2023).