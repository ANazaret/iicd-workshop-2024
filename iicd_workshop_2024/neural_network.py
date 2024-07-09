import torch


class DenseNN(torch.nn.Module):
    """
    A simple feedforward neural network with ReLU activation function.

    Parameters
    ----------
    n_input : int
        The number of input features.
    n_output : int
        The number of output features.
    n_hidden : int
        The number of hidden units in each hidden layer.
    n_layers : int
        The number of hidden layers
    """

    def __init__(self, n_input: int, n_output: int, n_hidden: int = 128, n_layers: int = 1):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(n_input, n_hidden))
        for _ in range(n_layers - 1):
            self.layers.append(torch.nn.Linear(n_hidden, n_hidden))
        self.layers.append(torch.nn.Linear(n_hidden, n_output))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)
