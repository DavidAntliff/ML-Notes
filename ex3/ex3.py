#!/usr/bin/env python
#
# Note: this uses the scipy.minimize(Truncated Newton) algorithm
# whereas the course uses an included fmincg function.
# The results are therefore different.


import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.optimize import minimize
import logging


logger = logging.getLogger(__name__)


def display_data(X, example_width=None):
    """Display 2D data in a nice grid.
    h, display_array = display_data(X, example_width) displays 2D data
    stored in X in a nice grid. It returns the figure handle h and the
    displayed array if requested.
    """

    if example_width is None:
        example_width = round(np.sqrt(X.shape[1]))
    example_width = int(example_width)

    # TODO: colormap

    # compute rows, cols
    m, n = X.shape
    example_height = int(n / example_width)

    # compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    # padding between images
    pad = 1

    # set up blank display
    display_array = -np.ones((int(pad + display_rows * (example_height + pad)),
                             int(pad + display_cols * (example_width + pad))))

    # copy each example into a patch in the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex >= m:
                break

            # copy patch
            max_val = max(abs(X[curr_ex, :]))
            # display_array[pad + j * (example_height + pad) + np.arange(example_height),
            #               pad + i * (example_width + pad) + np.arange(example_width)] = \
            #                   np.reshape(X[curr_ex, :], (example_height, example_width)) / max_val;
            ix = np.ix_(pad + j * (example_height + pad) + np.arange(example_height),
                        pad + i * (example_width + pad) + np.arange(example_width))
            display_array[ix] = np.reshape(X[curr_ex, :], (example_height, example_width)) / max_val;
            curr_ex += 1

        if curr_ex >= m:
            break

    # transpose for imshow
    display_array = display_array.T

    # flip the image
    display_array = np.flipud(display_array)

    # display image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    h = ax.imshow(display_array, origin="lower", cmap="gray")
    plt.show(block=False)

    return h, display_array


def sigmoid(z):
    """Compute sigmoid function."""
    return 1.0 / (1 + np.exp(-z))


def calc_J(theta, X, y, lambda_):
    m = len(y)
    h = sigmoid(X.dot(theta))
    t = -y * np.log(h) - (1 - y) * np.log(1 - h)
    J = sum(t) / m
    return J[0]

def calc_grad(theta, X, y, lambda_):
    m = len(y)
    h = sigmoid(X.dot(theta))
    error = h - y
    reg = lambda_ / m * theta
    reg[0] = 0
    grad = X.T.dot(error) / m + reg
    return grad


def lr_cost_function(theta, X, y, lambda_):
    """Compute cost and gradient for logistic regression with regularization.
    J, grad = lr_cost_function(theta, X, y, lambda_) computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. the parameters.
    """
    J = calc_J(theta, X, y, lambda_)
    grad = calc_grad(theta, X, y, lambda_)
    return J, grad


def one_vs_all(X, y, num_labels, lambda_):
    """Trains multiple logistic regression classifiers and returns all
    the classifiers in a matrix all_theta, where the i-th row of all_theta
    corresponds to the classifier for label i.
    all_theta = one_vs_all(X, y, num_labels, lambda) trains num_labels
    logistic regression classifiers and returns each of these classifiers
    in a matrix all_theta, where the i-th row of all_theta corresponds
    to the classifier for label i.
    """
    m, n = X.shape  # n is num_parameters

    # add column of ones to the X data matrix
    X = np.c_[np.ones(m), X]

    # pre-create all_theta
    all_theta = np.zeros((num_labels, n + 1))

    def cost(theta, X, y, lambda_):
        # must reshape theta to (N, 1):
        theta = np.reshape(theta, (-1, 1))
        J = calc_J(theta, X, y, lambda_)
        return J

    def gradient(theta, X, y, lambda_):
        # must reshape theta to (N, 1):
        theta = np.reshape(theta, (-1, 1))
        # must flatten result to (N, ) array for minimize()
        grad = calc_grad(theta, X, y, lambda_)
        #return np.ndarray.flatten(grad)
        return grad.ravel()  # ravel() is slightly faster

    for k in range(1, num_labels + 1):
        logger.info(f"Training for k == {k}")
        initial_theta = np.zeros((n + 1, 1))
        y_i = (y == k).astype(int)
        fmin = minimize(fun=cost,
                        x0=initial_theta,
                        args=(X, y_i, lambda_),
                        method='TNC',  # seems faster and better than CG or BFGS
                        #options={'maxiter': 1000},
                        jac=gradient)
        logger.info(f"Cost: {fmin.fun}")
        logger.info(f"Iterations: {fmin.nit}")
        all_theta[k - 1, :] = fmin.x

    return all_theta


def predict_one_vs_all(all_theta, X):
    """Predict the label for the trained one-vs-all classifier. The labels
    are in the range 1..K where K = size(all_theta, 1).
    p = predict_one_vs_all(all_theta, X) will return a vector of predictions
    for each example in the matrix X. Note that X contains the examples in
    rows. all_theta is a matrix where the i-th row is a trained logistic
    regression theta vector for the i-th class.
    """

    # add column of ones to the X data matrix
    m = X.shape[0]
    X = np.c_[np.ones(m), X]

    h = sigmoid(X.dot(all_theta.T))
    h_argmax = np.argmax(h, axis=1) + 1
    return np.reshape(h_argmax.T, (-1, 1))


def main():
    logging.basicConfig(level=logging.INFO)

    input_layer_size = 400  # 20x20 input images of digits
    num_labels = 10         # digits 0-9 (0 is mapped to 10)

    data = scipy.io.loadmat('data/ex3data1.mat')
    X = data["X"]
    y = data["y"]
    m = X.shape[0]

    # randomly select 100 data points to display
    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[:100], :]

    display_data(sel)

    # Part 2a: Vectorize Logistic Regression
    print("Testing lr_cost_function() with regularization")
    theta_t = np.array([[-2, -1, 1, 2]]).T
    X_t = np.c_[np.ones((5, 1)), np.reshape(np.arange(1, 16), (5, 3), order='F') / 10.]
    y_t = (np.array([[1, 0, 1, 0, 1]]).T >= 0.5).astype(int)
    lambda_t = 3
    J, grad = lr_cost_function(theta_t, X_t, y_t, lambda_t)

    print(f"Cost: {J}")
    print(f"Expected cost: 2.534819")
    print(f"Gradients:")
    print(f" {grad}")
    print(f"Expected gradients:")
    print(f" 0.146561, -0.548558, 0.724722, 1.398003")

    # Part 2b: One-vs-All Training
    lambda_ = 0.1
    all_theta = one_vs_all(X, y, num_labels, lambda_)

    # Part 3: predict for One-vs-All
    pred = predict_one_vs_all(all_theta, X)
    accuracy = np.mean((pred == y.astype(int))) * 100
    print(f"Training set accuracy: {accuracy}%")

    plt.show()


if __name__ == "__main__":
    main()
