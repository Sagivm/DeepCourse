import numpy as np
def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


def to_one_hot(y, num_classes):
    # convert a probability vector to a one-hot encoded vector

    one_hot = np.zeros((len(y), num_classes))

    for i in range(len(y)):
        one_hot[i, y[i]] = 1

    return one_hot