# Problem set 3 - Probabilistic modeling
Wednesday, 2024-07-10

---

In this problem set, we explore probabilistic modeling on scRNA-sequencing data.

You can find the solutions to this problem set in this notebook: [problem_set_3.ipynb](https://github.com/ANazaret/iicd-workshop-2024/blob/main/problem-set-3.ipynb).

We recommend starting a new conda environment for this problem set.
```bash

conda create -n iicd-workshop python=3.11
conda activate iicd-workshop
```

Before starting, make sure the `iicd-workshop-2024` package is installed, and that you have the
latest version of it.
```bash
pip install iicd-workshop-2024 --upgrade
```

As well as other dependencies:
```bash
pip install scvi-tools torch scanpy seaborn matplotlib
```


## Problem 1 - Simple model with global gene parameters

In this problem, you will implement simple models with global gene parameters.
For each cell $i$ and gene $g$, assume that the gene expression $x_{i,g}$ follows a distribution
$$ x_{i,g} \sim p(\theta_g) $$
where $\theta_g$ contains gene specific parameters shared across cells, and $p$ is a distribution (e.g. Normal, Negative Binomial).


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

You will implement simple gene model by subclassing the `BaseGeneModel` class provided
in the `iicd_workshop_2024` package. Link to the documentation: [BaseGeneModel](references.md#iicd_workshop_2024.gene_model.BaseGeneModel)
This class already implements the `fit` and `loss` methods for you.
You just need to implement the `get_distribution` method that returns the distribution of the model,
as well as the `__init__` method that initializes the model.

Here is a template for the `NormalGeneModel` class:
```python
import torch.distributions as dist
from iicd_workshop_2024.gene_model import BaseGeneModel


class NormalGeneModel(BaseGeneModel):
    def __init__(self, n_genes: int):
        super().__init__(n_genes)
        # declare any parameters here, using torch.nn.Parameter
        ...

    def get_distribution(self, gene_idx=None) -> dist.Distribution:
        # return the distribution for the gene_idx-th gene
        return ...

```

#### 2.1) Implement the `__init__` and the `get_distribution` methods

You should use the `dist.Normal` class from `torch.distributions` to create the distribution object.
Here is the documentation for [torch.distributions](https://pytorch.org/docs/stable/distributions.html).


#### 2.2) Fit the model

Fit the model to the data using the `fit` method of the `NormalGeneModel` class.
We recommend you to read the documentation and source code for the [fit](references.md/#iicd_workshop_2024.inference.fit) method.

#### 2.3) Visualize the learned means

Visualize the learned gene means against the true empirical gene means.

- Are they the same?
- If not, can you hypothesize reasons for any differences?
- And can you fix it?

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

#### 2.4) Visualize a few gene distributions

Visualize a few gene distributions by plotting the learned distributions against the empirical distribution.
For example, genes: `["CD14", "CD74", "RPS27"]`.

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
where:
$f$ is a function that transforms the cell specific representation into the gene specific parameters.

For this problem, we define the family of distributions $p$ to be negative binomial distributions

We further will use amortized inference to learn the cell specific representations $z_i$.
$$ z_i \approx g(x_i), $$
where $g$ is a neural network that takes the gene expression $x_i$ as input and outputs the cell specific representation $z_i$.

### 1) Implement the auto-encoder model

Implement the auto-encoder model by completing the following class.
You may use the [fit](references.md/#iicd_workshop_2024.inference.fit) function to train the model
as is done in the [BaseGeneModel](references.md#iicd_workshop_2024.gene_model.BaseGeneModel).
```python
import torch


class LatentModel(torch.nn.Module):
    def __init__(self, n_genes: int, n_latent: int):
        super().__init__()
        ...

    def get_distribution(self, data):
       ...

    def loss(self, data):
        return -self.forward(data).log_prob(data).mean()
```

You can also use the [DenseNN](references.md/#iicd_workshop_2024.neural_network.DenseNN)
class from the `iicd_workshop_2024.neural_network` module to define the neural network.
```python
from iicd_workshop_2024.neural_network import DenseNN
```


### 2) Fit the model
You can use the `fit` function from the `iicd_workshop_2024.inference` module to fit the model.

### 3) Implement a `get_latent_representation` method
This function should be able to retrieve the latent vectors `z` for any given `x` input.

### 4) Visualize the learned cell specific representations using UMAP
You can use `scanpy` to visualize the learned cell specific representations using UMAP.
Does the latent space appear coherent? Can you validate whether the latent space preserves any prior annotations expected to dominate the signal?

### 5) Compare against Decipher
We would now like to see how our simple autoencoder model compares to Decipher.

Install `decipher` by running:
```bash
pip install scdecipher
```

Decipher has an API close to `scanpy`, with computational function in `decipher.tl` and plotting functions in `decipher.pl`.
You can fit a Decipher model to the data using the following code snippet:
```python
import decipher as dc

# color by cell type (called 'str_labels' in the pbmc dataset)
model = dc.tl.decipher_train(adata, plot_every_k_epochs=1, plot_kwargs=dict(color='str_labels'))
```

You should be able to train the model and retrieve a similar latent representation.
Visualize this representation and compare it against the one from your autoencoder.

How do they differ?

Optional:

- Decode a series of points across a data-dense region of the latent representations of the auto-encoder and Decipher.
- Then, for each gene (or a select few genes), plot the trend in gene expression values corresponding to the series of points.
- Do they appear smooth or discontinuous?
- Are there correlations between certain genes?
- Are there sudden shifts in gene expression that correspond with annotation changes?

## Problem 3 - Implement your own model

Now you can implement your own model on your own data. Good luck!
