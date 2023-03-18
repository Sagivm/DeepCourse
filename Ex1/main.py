# IMPORTS
import numpy as np


def initialize_parameters(layers_dims: list) -> dict:
    """
    Create an ANN architecture depending on layers_dims
    :param layers_dims: list of layers dimentions
    :type layers_dims: list
    :return: dictionary built as follows:
        W: list of matrices representing layer's weights, initialized randomly,
        b: list of biases for each layer, initialized to zero
    :rtype: dict
    """
    # Create W

    W_sizes = [(layers_dims[i], layers_dims[i + 1]) for i, _ in enumerate(layers_dims[:-1])]
    W = [np.random.randn(*Wi_size) for Wi_size in W_sizes]

    # create b

    b = np.zeros(len(layers_dims) - 1)

    return {
        "W": W,
        "b": b
    }


def linear_forward(A: np.ndarray, W: np.ndarray, B: np.ndarray) -> dict:
    """
    Performing linear forward on NN
    :param A: Activation vector of previous layer
    :type A: np.ndarray
    :param W: Weight matrix of the current layer
    :type W: np.ndarray
    :param B: Bias vector of the current layer
    :type B: np.ndarray
    :return: dictionary built as follows:
        Z: linear component of activation function
        linear_cache: A,W,B
    :rtype: dict
    """
    return {
        "Z": np.transpose(W) * A + B,
        "linear_cache": {
            "A": A,
            "W": W,
            "B": B
        }
    }


def softmax(Z: np.ndarray) -> dict:
    """
    Applying softmax on Z
    :param Z: the linear component of the activation function
    :type Z: np.ndarray
    :return: dictionary built as follows:
        A: Activation of th layer
        activation_cache: Z
    :rtype: dict
    """
    Z_sum = np.sum(np.exp(Z))
    return {
        "A": np.exp(Z) / Z_sum,
        "activation_cahce": Z
    }


def relu(Z: np.ndarray) -> dict:
    """
        Applying relu on Z
        :param Z: the linear component of the activation function
        :type Z: np.ndarray
        :return: dictionary built as follows:
            A: Activation of th layer
            activation_cache: Z
        :rtype: dict
        """
    return {
        "A": np.maximum(0, Z),
        "activation_cahce": Z
    }


if __name__ == "__main__":
    x = initialize_parameters([3, 4, 5])
    # linear_forwward()
