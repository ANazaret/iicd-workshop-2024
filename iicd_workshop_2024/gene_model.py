import abc

import numpy as np
import seaborn as sns
import torch
import torch.distributions as dist
from matplotlib import pyplot as plt

from iicd_workshop_2024.inference import fit


class BaseGeneModel(abc.ABC, torch.nn.Module):
    """
    Base class for modeling the expression of gene expression using distributions with
    gene-specific parameters that are shared across cells.

    Args:
        n_genes (int): The number of genes to model.
    """

    def __init__(self, n_genes):
        super().__init__()
        self.n_genes = n_genes

    @property
    def distribution_name(self):
        """
        Get the name of the distribution used for modeling the gene expression.
        """
        return self.get_distribution().__class__.__name__.lower()

    @abc.abstractmethod
    def get_distribution(self, gene_idx=None) -> dist.Distribution:
        """
        Get the distribution that models the gene expression.

        Args:
            gene_idx (int or list[int] or None): If None, return the distribution over all genes. Otherwise, return the distribution
                of the specified gene or list of genes (given by their indices).

        Returns:
            dist.Distribution or list[dist.Distribution]: The distribution(s) of the gene(s).
        """
        pass

    def loss(self, data) -> torch.Tensor:
        """
        Return the negative log-likelihood of the data given the model.

        Returns:
            torch.Tensor: The negative log-likelihood of the data given the model.
        """
        return -self.get_distribution().log_prob(data).mean()

    def fit(self, adata, epochs=100, batch_size=128, lr=1e-2):
        """
        Fit the model to the data.

        Args:
            adata (AnnData): Annotated data matrix.
            epochs (int): Number of epochs to train the model.
            batch_size (int): Batch size.
            lr (float): Learning rate.
        """
        fit(self, adata, epochs=epochs, batch_size=batch_size, lr=lr)


def plot_gene_distribution(model: BaseGeneModel, adata, genes, n_cols=3):
    """
    Plot the learned distributions and the empirical distributions of the genes.

    Args:
        model (BaseGeneModel): The gene model.
        adata (AnnData): The annotated data matrix.
        genes (list[str]): The list of genes to plot.
        n_cols (int): The number of columns in the plot.
    """
    n_rows = int(np.ceil(len(genes) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), squeeze=False)
    for i, gene in enumerate(genes):
        ax = axs[i // n_cols, i % n_cols]
        gene_idx = adata.var["gene_symbols"].tolist().index(gene)
        sns.histplot(adata.X[:, gene_idx].toarray(), stat="density", discrete=True, ax=ax)
        max_value = adata.X[:, gene_idx].max().item()
        if model.distribution_name in ["poisson", "negativebinomial"]:
            x = torch.arange(0, max_value + 1)
        else:
            x = torch.linspace(
                min(
                    -5,
                    model.get_distribution(gene_idx).mean.item()
                    - 2 * model.get_distribution(gene_idx).stddev.item(),
                ),
                max_value,
                200,
            )
        y = model.get_distribution(gene_idx).log_prob(x).exp().detach().numpy()
        sns.lineplot(x=x, y=y, ax=ax, color="red")
        ax.set_title(gene + f" (idx={gene_idx})")
    plt.tight_layout()
    plt.show()
