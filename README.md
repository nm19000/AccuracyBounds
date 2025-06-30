# Accuracy Bounds

## Accuracy Bound for Inverse Problems

Computation of worst-case and average kernel size from [1] for an inverse problem with noise of the form: 

$$
\text{Given measurements } y = F(x,e)=Ax+e \text{ of } x \in \mathcal{M}_1 \subset \mathbb{R}^N \text{ and } e \in \mathcal{E} \subset \mathbb{R}^m, \text{ recover } x.
$$

Initial algoritms converge under the assumption that 
$$
\pi_1(P_{\mathcal{N}(F)^\perp}(x,e) - P_{\mathcal{N}(F)}(x,e)) \in \mathcal{M}_1.
$$

### Algoritm for computing the worst-case kernel size

1) Randomly with any distribution $\mu$ sample 
$$
\{y_i\}_{i=1}^k \subset \mathcal{M}_2 = A(\mathcal{M}_1)
$$
for $k \in \mathbb{N}$. For incresing $k$ only new $y_i \in \mathcal{M}_2$ are added and the old sampled points are kept.
2) Compute 
$$
\text{diam}(\pi_1(F_{y_i})) = \sup_{x,x' \in \pi_1(F_{y_i})} d_1(x,x')
$$ 
using a finite approximation $F_{y_i}^n \subseteq F_{y_i}$ with $|F_{y_i}^n|=n$, where $n = n(k)$ is a function of $k$ such that $n \geq k$ and $n \to \infty$ when $k \to \infty$. 
Set $F^0_{y_i} = \emptyset$ and 
$$
x^\perp_i = \pi_1(F^\dagger y_i) \in P_{\mathcal{N}(F)^\perp}(\mathcal{M}_1).
$$ 
Then, iteratively for $n \in \mathbb{N}$ for randomly sampled $x_n \in \mathcal{M}_1$, if
       $$
        e_n:= y_i - F(x_n,0)\in \mathcal{E},
       $$
    let $F^n_{y_i}  = F^{n-1}_{y_i}  \cup \{x_n\}$ with the $x_n$ chosen as above. 
    Set $\text{diam}(\pi_1(F^0_{y_i}))=0$. Then, for each $n \in \mathbb{N}$ if 
        $$
        ||\pi_1(P_{\mathcal{N}(F)}(x_n,e_n))|| = ||\pi_1((I-F^{\dagger}F)(x_n,e_n))|| > ||\pi_1((I-F^{\dagger}F)(x_{n-1},e_{n-1}))||,
        $$
    set 
    $$
    \text{diam}(\pi_1(F^n_{y_i}))= 2||\pi_1((I-F^{\dagger}F)(x_n,e_n))||
    $$ 
    and if 
        $$
        ||\pi_1(P_{\mathcal{N}(F)}(x_n,e_n))|| = ||\pi_1((I-F^{\dagger}F)(x_n,e_n))|| \leq  ||\pi_1((I-F^{\dagger}F)(x_{n-1},e_{n-1}))||,
        $$
    set 
    $$
    \text{diam}(\pi_1(F^n_{y_i}))= 2||\pi_1((I-F^{\dagger}F)(x_{n-1},e_{n-1}))||.
    $$
3) Now obtain the approximate worst case kernel size by 
       $$
       \text{kersize}^w(\mathcal{M}_1, A,\mathcal{E} )_{k} = \max_{i \in \{1, ..., k\}} \text{diam}(\pi_1(F^{n(k)}_{y_i})).
       $$

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
    ||v_n^i||= ||\pi_1((I-F^{\dagger}F)(x_n,e_n))||.
$$

3) Now obtain the approximate average case kernel size by 
$$
\text{ker}^a(F, \mathcal{M}_1, \mathcal{E},p)_{k}^p = (F_*\mu)(\mathcal{M}_2) 2^p/k \sum_{i=1}^k 1/n(i) \sum_{l=1}^{n(i)} || v_l^i ||^p.
$$
4) For any $\delta >0$, obtain confidence intervals $c(k,\delta) = c(k,\delta,F, \mathcal{M}_1, \mathcal{E},p)$ with probability at least $1-c(k,\delta)$ using Hoeffding's inequality for exchangeable random variables.

# References
[1] Gottschling, Nina Maria, et al. "On the existence of optimal multi-valued decoders and their accuracy bounds for undersampled inverse problems." arXiv preprint arXiv:2311.16898 (2023).