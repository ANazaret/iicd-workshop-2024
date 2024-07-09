import scipy
import torch
import tqdm


def fit(model, adata, epochs=100, batch_size=128, lr=1e-2):
    """
    Fit the model to the data.

    Parameters
    ----------
    model:
        The model to fit.
    adata: AnnData
        The annotated data matrix.
    epochs: int
        Number of epochs to train the model.
    batch_size: int
        Batch size.
    lr: float
        Learning rate.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data_X = adata.X
    # check if sparse
    if isinstance(data_X, scipy.sparse.csr_matrix):
        data_X = data_X.toarray()
    data_loader = torch.utils.data.DataLoader(data_X, batch_size=batch_size, shuffle=True)
    pbar = tqdm.tqdm(total=epochs * len(data_loader))
    for _ in range(epochs):
        for x in data_loader:
            optimizer.zero_grad()
            loss = model.loss(x).mean()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
            pbar.update()
    pbar.close()
