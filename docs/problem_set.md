# Problem set 3 - Probabilistic modeling
Wednesday, 2024-07-10

---

In this problem set, we will explore probabilistic modeling on scRNA-sequencing data.

## Problem 1 - Simple model with global gene parameters

In this problem, you will implement simple models with global gene parameters.
Assume that for each cell $i$ and gene $g$, the gene expression $x_{i,g}$ is distributed according to a
distribution
$$ x_{i,g} \sim p(\theta_g) $$
where $\theta_g$ contains gene specific parameters shared across cells, and $p$ is a distribution (e.g. Normal, Negative Binomial).

You will implement simple gene model by subclassing the class  `BaseGeneModel` provided
in the `iicd_workshop_2024` package.

Link to the documentation: [BaseGeneModel](references.md#iicd_workshop_2024.gene_model.BaseGeneModel)

### 1) Load the data

Load public scRNA-seq data `pbmc` from the `scvi-tools` package.
```python
import scvi

adata = scvi.data.pbmc_dataset()
```

### 2) Implement a simple Normal gene model

In this question, assume that
$$ x_{i,g} \sim \mathcal{N}(\mu_g, \sigma_g^2) $$
where $\mu_g$ and $\sigma_g$ are gene specific parameters shared across cells.

Subclass the `BaseGeneModel` class to implement Normal gene model.

```python
import torch.distributions as dist
from iicd_workshop_2024.gene_model import BaseGeneModel


class NormalGeneModel(BaseGeneModel):
    def get_mean(self, gene_idx=None):
        if gene_idx is None:
            gene_idx = slice(None)
        return ...

    def get_std(self, gene_idx=None):
        if gene_idx is None:
            gene_idx = slice(None)
        return ...

    def get_distribution(self, gene_idx=None) -> dist.Distribution:
        return ...

```

#### 2.1) Implement the `get_mean` method

#### 2.2) Implement the `get_std` method

#### 2.3) Implement the `get_distribution` method

You should use the `dist.Normal` class from `torch.distributions` to create the distribution.

#### 2.4) Fit the model

Fit the model to the data using the `fit` method.

#### 2.5) Visualize the learned distributions

- Visualize the learned gene means against the true empirical gene means.
- Are they the same? Do you have any ideas why they might differ?
- If they differ and you have an idea why, can you fix it?

To visualize the gene means, you can use the following code snippet:
```python
import seaborn as sns
import matplotlib.pyplot as plt


def plot_learned_vs_empirical_mean(model, adata):
    empirical_means = ...
    learned_means = ...
    sns.scatterplot(x=empirical_means, y=learned_means)
    max_value = max(empirical_means.max(), learned_means.max())
    plt.plot([0, max_value], [0, max_value], color='black', linestyle='--')
    plt.xlabel("Empirical mean")
    plt.ylabel("Learned mean")
    plt.show()
```

#### 2.6) Visualize a few gene distributions

Visualize a few gene distributions by plotting the learned distributions against the empirical distribution.

You can use the function `plot_gene_distribution` from the `iicd_workshop_2024.gene_model` module.
See documentation for [plot_gene_distribution](references.md/#iicd_workshop_2024.gene_model.plot_gene_distribution).


### 3) Now repeat the same steps for a Poisson gene model

Math reminder:

- The Poisson distribution is a distribution over non-negative integers.
- It is parametrized by a single parameter $\lambda$, its mean (which is also its variance).
$$ x \sim \text{Poisson}(\lambda) $$
with $\mathbb{E}[x] = \text{Var}[x] = \lambda$.



### 4) Now repeat the same steps for a Negative Binomial gene model
Math reminder:

- The Negative Binomial distribution is a distribution over non-negative integers.
- It is parametrized by two parameters: the mean $\mu$ and the inverse dispersion parameter $\alpha$.
$$ x \sim \text{NegBinom}(\mu, \alpha) $$ with
$ \mathbb{E}[x] = \mu$ and $\text{Var}[x] = \mu + \frac{\mu^2}{\alpha}$.
- The Negative Binomial distribution implemented in pytorch does not use the mean and inverse dispersion parameter.
  Instead, it uses the `total_count` and `logit` parameters.
  The correspondance is given by:
  - total_count = inverse dispersion
  - logit = log(mean) - log(inverse dispersion)

## Problem 2 - Gene model with cell specific parameters

In this problem, you will implement gene models with cell specific parameters, learned with an
autoencoder.

In the previous problem, we assumed that the gene expression $x_{i,g}$ was distributed according to:
$$ x_{i,g} \sim p(\theta_g) $$
where $\theta_g$ contains gene specific parameters shared across cells.
In reality, the gene expression is influenced by cell specific parameters as well.

The idea of representation learning for high-dimensional data is that we might not
know exactly what are the cell specific parameters that influence the gene expression,
but we assume that there exist a small number of them.

We write $z_i$ the cell specific representation and we obtain the model:
$$
\begin{align}
x_{i,g} &\sim p(f(z_i), \theta_g)
\end{align}
$$
$$
\begin{align}
z_i \sim \pi, \quad \theta_g \sim \rho
\end{align}
$$
where:

- $\pi$ is the prior distribution of the cell specific representations,
- $\rho$ is the prior distribution of the gene specific parameters
- $f$ is a function that transforms the cell specific representation into the gene specific parameters.

For this problem, we set:

- the family of distributions $p$ to be a negative binomial distributions
- the prior distribution of the gene specific parameters $\rho$ to be uniform
- the prior distribution of the cell specific representations $\pi$ to be a standard normal distribution.
