#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import logging
import ex3


logger = logging.getLogger(__name__)


def predict(Theta1, Theta2, X):
    """Predict the label of an input given a trained neural network
    p = predict(Theta1, Theta2, X) outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2).
    """
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    a1 = np.c_[np.ones((m, 1)), X]
    z2 = a1.dot(Theta1.T)
    a2 = np.c_[np.ones((m, 1)), ex3.sigmoid(z2)]
    z3 = a2.dot(Theta2.T)
    a3 = ex3.sigmoid(z3)

    h_argmax = np.argmax(a3, axis=1) + 1
    return np.reshape(h_argmax.T, (-1, 1))


def main():
    logging.basicConfig(level=logging.INFO)

    input_layer_size = 400  # 20x20 input images of digits
    hidden_layer_size = 25  # 25 hidden units
    num_labels = 10         # digits 0-9 (0 is mapped to 10)

    data = scipy.io.loadmat('data/ex3data1.mat')
    X = data["X"]
    y = data["y"]
    m = X.shape[0]

    # randomly select 100 data points to display
    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[:100], :]

    ex3.display_data(sel)

    # load pre-calculated weights
    weights = scipy.io.loadmat('data/ex3weights.mat')
    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']

    # Part 3: NN predict
    pred = predict(Theta1, Theta2, X)
    accuracy = np.mean((pred == y.astype(int))) * 100
    print(f"Training set accuracy: {accuracy}%")

    plt.show()


if __name__ == "__main__":
    main()
