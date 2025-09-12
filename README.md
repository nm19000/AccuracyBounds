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
            - $F_{y_k}^{N} \gets F_{y_k}^{N} \bigcup x_{k,n}$
            - $\mathcal{D} \gets \mathcal{D} \bigcup (x_{k,n}, y_k)$
        - ElseIf $|F_{y_k}^{N}| \geq N(K)_{max}$:
            - Break 
        - EndIf 
    - EndFor
    - $N(k) = |F_{y_k}^{N}|$
    - $(N(k))_{k=1}^K \gets \bigcup_k N(k)$
    - $(F_{y_k}^{N})_{k=1}^K \gets \bigcup_k F_{y_k}^{N(k)}$
- EndFor 
- Return:  $(F_{y_k}^{N})_{k=1}^K$, $(N(k))_{k=1}^K$, $\mathcal{D}$

### Algoritm for Computing the Worst-Case Kernel Size From Feasible Sets

- Require $p \in (0,\infty)$, $K \in \mathbb{N}$, $(F_{y_k}^{N})_{k=1}^K$, $(N(k))_{k=1}^K$
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

- Require $p \in (0,\infty)$, $K \in \mathbb{N}$,  $(F_{y_k}^{N(k)})_{k=1}^K$, $(N(k))_{k=1}^K$
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
[1] Gottschling, Nina Maria, et al. "On the existence of optimal multi-valued decoders and their accuracy bounds for undersampled inverse problems." arXiv preprint arXiv:2311.16898 (2023).