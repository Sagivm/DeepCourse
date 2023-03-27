# IMPORTS
from typing import Callable

import numpy as np
# from util import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


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

    W_sizes = [(next_dim, current_dim) for current_dim, next_dim in zip(layers_dims[:-1], layers_dims[1:])]
    W = [np.random.randn(*Wi_size) for Wi_size in W_sizes]

    # create b

    b_sizes = layers_dims[1:]
    b = [np.zeros((1, bi_size)) for bi_size in b_sizes]

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
        "Z": A.dot(W.T) + B,
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
    return {
        "A": np.exp(Z) / np.exp(Z).sum(),
        "activation_cahce": {
            "Z": Z
        }
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
        "activation_cahce": {
            "Z": Z
        }
    }


def linear_activation_forward(A_prev: np.ndarray, W: np.ndarray, B: np.ndarray,
                              activation: Callable[[np.ndarray], dict]) -> dict:
    cache = {}
    linear = linear_forward(A_prev, W, B)
    z, linear_cache = linear['Z'], linear['linear_cache']

    active = activation(z)
    a, activation_cache = active['A'], active['activation_cache']

    cache.update(linear_cache)
    cache.update(activation_cache)

    return {
        "A": a,
        "cache": cache
    }


def L_model_forward(X: np.ndarray, parameters: dict, use_batchnorm: bool = False):
    """

    :param X: matrix of inputs
    :type X: np.ndarray
    :param parameters: a dict like object containing W and b
    :type parameters: dict
    :param use_batchnorm: whether to use batch normalization or not
    :type use_batchnorm: bool
    :return:
        dictionary containing the activation of the ANN represented by the parameters on X and cache actions
    :rtype:
        dict
    """
    cache_list = list()
    A = X

    # Relu layers
    for W_i, b_i in zip(parameters["W"][:-1], parameters["b"][:-1]):
        results = linear_activation_forward(A, W_i, b_i, relu)
        A = results['A']
        if use_batchnorm:
            raise NotImplementedError()

        cache_list.append(results['cache'])

    # Softmax layer
    cache = {}
    Z, cache["linear_cache"] = list(linear_forward(A, parameters["W"][-1], parameters["b"][-1]).values())
    y, cache["activation_cache"] = list(softmax(Z).values())
    cache_list.append(cache)
    return y, cache_list


def compute_cost(Al: np.ndarray, Y: np.ndarray):
    """
    Compute loss(cost) using prediction(Al) and true values(Y)
    :param Al:
    :type Al:
    :param Y:
    :type Y:
    :return:
    :rtype:
    """
    return np.sum(Y + 1 * np.log(Al + 1)) / Y.shape[0]  # +1 to avoid log 0


def apply_batchnorm(A: np.ndarray) -> np.ndarray:
    NotImplementedError()


# Section 2


def linear_backward(dZ: np.ndarray, cache: dict):
    """
Implements the linear part of the backward propagation process for a single layer
    :param dZ: the gradient of the cost with respect to the linear output of the current laye
    :type dZ: np.ndarraty
    :param cache:
    :type cache: dict
    :return:
        tuple of derivatives dA,dW,dB
    :rtype:
    """
    dA = np.dot(dZ, cache["W"])
    dW = np.dot(cache['A'].T, dZ)
    dB = np.sum(dZ, axis=0, keepdims=True)
    return dA, dW, dB


def linear_activation_backward(dA: np.ndarray, cache: dict, activation):
    """
    Implements the backward propagation for the LINEAR->ACTIVATION layer. The function first computes dZ and then applies the linear_backward function.
    :param dA: post activation gradient of the current layer
    :type dA: np.ndarray
    :param cache: contains both the linear cache and the activations cache
    :type cache: dict
    :param activation: activation backward function
    :type activation: function
    :return:
                tuple of derivatives dA,dW,dB
    :rtype:
    """
    dZ = activation(dA, cache['activation_cache'])
    return linear_backward(dZ, cache['linear_cache'])


def relu_backward(dA: np.ndarray, activation_catch: dict):
    """
    Implements backward propagation for a ReLU unit
    :param dA: the post-activation gradient
    :type dA: np.ndarray
    :param activation_catch: contains Z (stored during the forward propagation)
    :type activation_catch: dict
    :return:
        derivative of Z
    :rtype:
        np.ndarray
    """
    dZ = np.array(dA, copy=True)
    dZ[activation_catch['Z'] <= 0] = 0
    return dZ


def softmax_backward(dA, activation_catch):
    # TODO: wtf
    return dA


def l_model_backward(Al: np.ndarray, Y: np.ndarray, caches: dict):
    """
    Implement the backward propagation process for the entire network.
    :param Al: the probabilities vector, the output of the forward propagation
    :type Al: np.ndarray
    :param Y: the true labels vector (the "ground truth" - true classifications)
    :type Y: np.ndarray
    :param caches: contains Z (stored during the forward propagation)
    :type caches: dict
    :return:
    gradient of the cost with respect to Z
    :rtype:
    np.ndarray
    """
    layers = len(caches) - 1
    grads = dict()

    # loss = compute_cost(Al, Y)  # maybe dz

    num_samples = len(Y)

    # TODO: understand score better
    
    ## compute the gradient on predictions
    dscores = Al.copy()
    dscores[range(num_samples), Y] -= 1
    dscores /= num_samples

    dA_curr = dscores

    # Layers update
    for i, layer in enumerate(reversed(caches)):
        grads[f"dA_{layers - i}"], grads[f"dW_{layers - i}"], grads[f"dB_{layers - i}"] = \
            linear_activation_backward(dA_curr, layer, softmax_backward) if i==0 else linear_activation_backward(grads[f"dA_{layers - i + 1}"], layer, relu_backward)
        dA = grads[f"dA_{layers - i}"]
    return grads


def update_parameters(parameters: dict, grads: dict, learning_rate: float):
    """
    Updates parameters using gradient descent
    :param parameters: parameters of the ANN
    :type parameters: dict
    :param grads: – a python dictionary containing the gradients (generated by L_model_backward)
    :type grads: dict
    :param learning_rate: the learning rate used to update
    :type learning_rate: float
    :return:
        Updated parameters of the ANN
    :rtype:
        dict
    """
    for index, _ in enumerate(parameters["W"]):
        parameters['W'][index] -= learning_rate * grads[f'dW_{index}'].T
        parameters['b'][index] -= learning_rate * grads[f'dB_{index}']
    return parameters


# For testing
if __name__ == "__main__":
    def get_data(path):
        data = pd.read_csv(path, index_col=0)

        cols = list(data.columns)
        target = cols.pop()

        X = data[cols].copy()
        y = data[target].copy()

        y = LabelEncoder().fit_transform(y)

        return np.array(X), np.array(y)


    X, Y = get_data(r'iris.csv')

    parameters = initialize_parameters([3, 6, 8, 10, 3])
    # X = np.random.randn(20, 3)
    # Y = np.random.randint(3, size=20)

    Y_hot = np.zeros((Y.size, np.max(Y) + 1))
    Y_hot[np.arange(Y.size), Y] = 1

    r, cache = L_model_forward(X, parameters)
    r_class = np.argmax(r, axis=-1)
    print(f"Acc - {accuracy_score(r_class, Y)} Cost - {compute_cost(r_class,Y)}")

    for i in range(1, 100):
        r_class = np.argmax(r, axis=-1)
        grads = l_model_backward(to_one_hot(r_class,3), Y, cache)
        parameters = update_parameters(parameters, grads, 0.01)
        r, cache = L_model_forward(X, parameters)
        print(f"Acc - {accuracy_score(np.argmax(r,axis=-1),Y)} Cost - {compute_cost(np.argmax(r, axis=-1),Y)}")
