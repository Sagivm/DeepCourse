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
        "Z": A.dot(W) + B,
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
    Z_sum = np.sum(np.exp(Z), axis=-1)
    return {
        "A": np.divide(np.transpose(np.exp(Z)), Z_sum).transpose(),
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


def L_model_forward(X: np.ndarray, parameters: dict, use_batchnorm: bool = False):
    """

    :param X: matrix of inputs
    :type X: np.ndarray
    :param parameters: a dict like object containing W and b
    :type parameters: dict
    :param use_batchnorm: if use batch or not
    :type use_batchnorm: bool
    :return:
        dictionary containing the activation of the ANN represented by the parameters on X and cache actions
    :rtype:
        dict
    """
    cache = list()
    A = X
    for W_i, b_i in zip(parameters["W"], parameters["b"]):
        cache.append(dict())
        Z, cache[-1]["linear_cache"] = list(linear_forward(A, W_i, b_i).values())
        if use_batchnorm:
            raise NotImplementedError()
        A, cache[-1]["activation_cache"] = list(relu(Z).values())

    cache.append(dict())
    y, cache[-1]["activation_cache"] = list(softmax(A).values())
    return y, cache


def compute_cost(Al: np.ndarray, Y: np.ndarray):
    pass


if __name__ == "__main__":
    parameters = initialize_parameters([5, 4, 2])
    X = np.random.randn(10, 5)
    r = L_model_forward(X, parameters)
    t = 0
