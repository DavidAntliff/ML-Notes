#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
import logging
import ex2
from scipy.optimize import fmin_bfgs


logger = logging.getLogger(__name__)


def map_feature(X1, X2):
    """Feature mapping function to polynomial features

    map_feature(X1, X2) maps the two input features
    to quadratic features used in the regularization exercise.

    Returns a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

    Inputs X1, X2 must be the same size
    """
    degree = 6
    out = np.ones(X1[:, 0].shape)
    for i in range(1, degree + 1):
        for j in range(i + 1):
            c = X1 ** (i - j) * (X2 ** j)
            out = np.c_[out, c]
    return out


def calc_J(theta, X, y, lambda_):
    m = len(y)
    h = ex2.sigmoid(X.dot(theta))
    t = -y * np.log(h) - (1 - y) * np.log(1 - h)
    J = sum(t) / m + lambda_ / (2 * m) * sum(theta[1: -1] ** 2)
    return J[0]


def calc_grad(theta, X, y, lambda_):
    m = len(y)
    h = ex2.sigmoid(X.dot(theta))
    error = h - y
    r = lambda_ / m * theta
    r[0] = 0
    grad = X.T.dot(error) / m + r
    return grad


def cost_function_reg(theta, X, y, lambda_):
    """Compute cost and gradient for logistic regression with regularization
    J = cost_function_reg(theta, X, y, lambda) computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    J = calc_J(theta, X, y, lambda_)
    grad = calc_grad(theta, X, y, lambda_)
    return J, grad


def plot_decision_boundary(theta, X, y, axes=None, colors=None):
    """Plots the data points X and y into a new figure with
    the decision boundary defined by theta
    plot_decision_boundary(theta, X,y) plots the data points with + for the 
    positive examples and o for the negative examples. X is assumed to be 
    MxN, N>3 matrix, where the first column is all-ones
    """
    if axes is None:
        ax = ex2.plot_data(X[:, (1, 2)], y)
    else:
        ax = axes

    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u), len(v)))

    # evaluate z = theta * X over the grid
    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = map_feature(np.array(u[i]).reshape((1, 1)),
                                  np.array(v[j]).reshape((1, 1))).dot(theta)
    z = z.T

    ax.contour(u, v, z, levels=[0], linewidth=2, colors=colors)
    return ax


def main():
    logging.basicConfig(level=logging.DEBUG)

    data = np.loadtxt('data/ex2data2.txt', delimiter=',')
    X = data[:, (0, 1)]
    y = data[:, 2].reshape((-1, 1))

    ax = ex2.plot_data(X, y)
    ax.set_xlabel('Microchip Test 1')
    ax.set_ylabel('Microchip Test 2')
    legend = ax.legend(labels=["y = 1", "y = 0"])

    # regularized logistic regression

    # add polynomial features
    X = map_feature(X[:, 0].reshape((-1, 1)), X[:, 1].reshape((-1, 1)))

    # all-zeros theta:
    initial_theta = np.zeros((X.shape[1], 1))
    lambda_ = 1

    cost, grad = cost_function_reg(initial_theta, X, y, lambda_)

    print(f"Cost at initial theta (zeros): {cost}")
    print(f"Expected cost (approx): 0.693")
    print(f"Gradient at initial theta (zeros) - first five values only:")
    print(f"{grad[0:5]}")
    print(f"Expected gradients (approx) - first five values only:")
    print(f" [ 0.0085 0.0188 0.0001 0.0503 0.0115 ... ]")

    # all-ones theta and lambda = 10
    test_theta = np.ones((X.shape[1], 1))
    cost, grad = cost_function_reg(test_theta, X, y, 10)

    print(f"Cost at test theta (with lambda = 10): {cost}")
    print(f"Expected cost (approx): 3.16")
    print(f"Gradient at test theta - first five values only:")
    print(f"{grad[0:5]}")
    print(f"Expected gradients (approx) - first five values only:")
    print(f" [ 0.3460 0.1614 0.1948 0.2269 0.0922 ... ]")

    # Regularization and Accuracies
    initial_theta = np.zeros((X.shape[1], 1))
    lambda_ = 1

    # closure wrappers for fmin_bgfs()
    def cost(theta):
        theta = theta.reshape((-1, 1))
        J = calc_J(theta, X, y, lambda_)
        logger.debug(f"J: {J}")
        return J
    def gradient(theta):
        theta = theta.reshape((-1, 1))
        #grad = np.ndarray.flatten(calc_grad(theta, X, y))
        grad = calc_grad(theta, X, y, lambda_).reshape((-1,))
        logger.debug(f"grad: {grad}")
        return grad  # must be (N,)

    theta = fmin_bfgs(cost, initial_theta, fprime=gradient)
    theta = theta.reshape((-1, 1))

    ax = plot_decision_boundary(theta, X, y)
    ax.set_title(f"lambda = {lambda_}")
    ax.set_xlabel("Microchip Test 1")
    ax.set_ylabel("Microchip Test 2")
    ax.legend(labels=["y = 1", "y = 0", "Decision Boundary"])

    p = ex2.predict(theta, X)

    print(f"Train Accuracy: {np.mean(p == y) * 100}")
    print(f"Expected accuracy (with lambda = 1): 83.1")

    # effects of varying lambda:
    lambdas = [0.0, 0.5, 1.0, 10.0, 100.0]
    ax = ex2.plot_data(X[:, (1, 2)], y)
    ax.legend(labels=['y = 1', 'y = 0'])

    handles, labels = ax.get_legend_handles_labels()

    # select a distinct colour for each contour plot
    def col_cycler(cols):
        count = 0
        while True:
            yield cols[count]
            count = (count + 1) % len(cols)
    col_iter = col_cycler(['r', 'y', 'g', 'b', 'c', 'm', 'k'])

    for lambda_ in lambdas:
        initial_theta = np.zeros((X.shape[1], 1))

        # closure wrappers for fmin_bgfs()
        def cost(theta):
            theta = theta.reshape((-1, 1))
            J = calc_J(theta, X, y, lambda_)
            logger.debug(f"J: {J}")
            return J
        def gradient(theta):
            theta = theta.reshape((-1, 1))
            #grad = np.ndarray.flatten(calc_grad(theta, X, y))
            grad = calc_grad(theta, X, y, lambda_).reshape((-1,))
            logger.debug(f"grad: {grad}")
            return grad  # must be (N,)

        theta = fmin_bfgs(cost, initial_theta, fprime=gradient)
        theta = theta.reshape((-1, 1))
        plot_decision_boundary(theta, X, y, axes=ax, colors=next(col_iter))

    ax.set_title("")
    ax.set_xlabel("Microchip Test 1")
    ax.set_ylabel("Microchip Test 2")

    proxy = [plt.Line2D((0, 0), (1, 1), color=pc.get_color()[0]) for pc in ax.collections]
    labels = [f"lambda = {str(x)}" for x in lambdas]
    ax.legend(proxy, labels)

    plt.show()


if __name__ == "__main__":
    main()
