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
    """

    def __init__(self, n_genes):
        super().__init__()
        self._mean = torch.nn.Parameter(torch.randn(n_genes))
        self._std = torch.nn.Parameter(torch.randn(n_genes))
        self._inverse_dispersion = torch.nn.Parameter(torch.randn(n_genes))

    @property
    def distribution_name(self):
        """
        Get the name of the distribution used for modeling the gene expression.
        """
        return self.get_distribution().__class__.__name__.lower()

    def get_mean(self, gene_idx=None):
        """
        Get the mean parameter of the distributions of gene.
        The method is used for Gaussian, Poisson, and negative binomial distributions.

        Parameters
        ----------
        gene_idx: int or list[int] or None
            If None, return the mean parameter of all genes. Otherwise, return the mean parameter
            of the specified gene or list of genes (given by their indices).

        Returns
        -------
        torch.Tensor or list[torch.Tensor]
            The mean parameter of the distribution(s) of the gene(s).
        """
        raise NotImplementedError

    def get_std(self, gene_idx=None):
        """
        Get the standard deviation parameter of the distributions of gene.
        The method is used for Gaussian distributions.

        Parameters
        ----------
        gene_idx: int or list[int] or None
            If None, return the standard deviation parameter of all genes. Otherwise, return the standard deviation parameter
            of the specified gene or list of genes (given by their indices).

        Returns
        -------
        torch.Tensor or list[torch.Tensor]
            The standard deviation parameter of the distribution(s) of the gene(s).

        """
        raise NotImplementedError

    def get_inverse_dispersion(self, gene_idx=None):
        """
        Get the inverse dispersion parameter of the distributions of gene.
        The method is used for negative binomial distributions.

        Parameters
        ----------
        gene_idx: int or list[int] or None
            If None, return the inverse dispersion parameter of all genes. Otherwise, return the inverse dispersion parameter
            of the specified gene or list of genes (given by their indices).

        Returns
        -------
        torch.Tensor or list[torch.Tensor]
            The inverse dispersion parameter of the distribution(s) of the gene(s).

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_distribution(self, gene_idx=None) -> dist.Distribution:
        """
        Get the distribution that models the gene expression.

        Parameters
        ----------
        gene_idx: int or list[int] or None
            If None, return the distribution over all genes. Otherwise, return the distribution
            of the specified gene or list of genes (given by their indices).

        Returns
        -------
        dist.Distribution or list[dist.Distribution]
            The distribution(s) of the gene(s).
        """
        pass

    def loss(self, data) -> torch.Tensor:
        """
        Return the negative log-likelihood of the data given the model.

        Parameters
        ----------
        data: torch.Tensor
            The observations on which to compute the negative log-likelihood.

        Returns
        -------
        torch.Tensor
            The negative log-likelihood of the data given the model.
        """
        return -self.get_distribution().log_prob(data).mean()

    def fit(self, adata, epochs=100, batch_size=128, lr=1e-2):
        """
        Fit the model to the data.

        Parameters
        ----------
        adata: AnnData
            Annotated data matrix.
        epochs: int
            Number of epochs to train the model.
        batch_size: int
            Batch size.
        lr: float
            Learning rate.
        """
        fit(self, adata, epochs=epochs, batch_size=batch_size, lr=lr)


def plot_gene_distribution(model: BaseGeneModel, adata, genes, n_cols=3):
    """
    Plot the learned distributions and the empirical distributions of the genes.

    Parameters
    ----------
    model: BaseGeneModel
        The gene model.
    adata: AnnData
        The annotated data matrix.
    genes: list[str]
        The list of genes to plot.
    n_cols: int
        The number of columns in the plot.
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
                min(-5, model.get_mean(gene_idx) - 2 * model.get_std(gene_idx)), max_value, 1000
            )
        y = model.get_distribution(gene_idx).log_prob(x).exp().detach().numpy()
        sns.lineplot(x=x, y=y, ax=ax, color="red")
        ax.set_title(gene + f" (idx={gene_idx})")
    plt.tight_layout()
    plt.show()
