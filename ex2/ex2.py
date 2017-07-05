#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs  # equivalent to Octave's fminunc()
import logging


logger = logging.getLogger(__name__)


def plot_data(X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pos = (y == 1)[:, 0]  # need to reshape to (N,)
    neg = (y == 0)[:, 0]  # need to reshape to (N,)
    ax.plot(X[pos, 0], X[pos, 1], 'k+', linewidth=2, markersize=7)
    ax.plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=7)
    return ax


def plot_decision_boundary(theta, X, y):
    """Plots the data points X and y into a new figure with
    the decision boundary defined by theta
    plot_decision_boundary(theta, X,y) plots the data points with + for the 
    positive examples and o for the negative examples. X is assumed to be 
    Mx3 matrix, where the first column is an all-ones column for the
    intercept.
    """
    ax = plot_data(X[:, (1, 2)], y)
    if X.shape[1] <= 3:
        # only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([min(X[:,1]) - 2, max(X[:,1]) + 2])

        # calculate the decision boundary line
        plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])

        ax.plot(plot_x, plot_y)
        ax.axis([30, 100, 30, 100])
    return ax


def sigmoid(z):
    """Compute sigmoid function."""
    return 1.0 / (1 + np.exp(-z))


def calc_J(theta, X, y):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    t = -y * np.log(h) - (1 - y) * np.log(1 - h)
    J = sum(t) / m
    return J[0]


def calc_grad(theta, X, y):
    m = len(y)
    h = sigmoid(X.dot(theta))
    error = h - y
    grad = X.T.dot(error) / m
    return grad


def cost_function(theta, X, y):
    """Compute cost and gradient for logistic regression
    J = cost_function(theta, X, y) computes the cost of using theta as the
    parameter for logistic regression and the gradient of the cost
    w.r.t. to the parameters.
    """
    J = calc_J(theta, X, y)
    grad = calc_grad(theta, X, y)
    return J, grad


def predict(theta, X):
    """Predict whether the label is 0 or 1 using learned logistic 
    regression parameters theta
    p = predict(theta, X) computes the predictions for X using a 
    threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    """
    return np.where(sigmoid(X.dot(theta)) >= 0.5, 1, 0)


def main():
    logging.basicConfig(level=logging.INFO)

    data = np.loadtxt('data/ex2data1.txt', delimiter=',')
    X = data[:, (0, 1)]
    y = np.reshape(data[:, 2], (-1, 1))

    ax = plot_data(X, y)
    ax.set_xlabel('Exam 1 score')
    ax.set_ylabel('Exam 2 score')
    ax.legend(labels=["Admitted", "Not Admitted"])

    # compute cost and gradient
    m, n = X.shape

    # add a column of ones to X for intercept term
    X = np.c_[np.ones(m), X]
    initial_theta = np.zeros((n + 1, 1))

    cost, grad = cost_function(initial_theta, X, y)
    print(f"Cost at initial theta (zeros): {cost}")
    print(f"Expected cost (approx): 0.693")
    print(f"Gradient at initial theta (zeros): {grad}")
    print(f"Expected gradients (approx): \n -0.1000\n -12.0092\n -11.2628\n")

    # optimizing using equivalent of Octave's fminunc()

    # use closure wrappers to condition theta correctly, and pass in X, y
    def cost(theta):
        # must reshape theta to (N,1):
        theta = np.reshape(theta, (-1, 1))
        J = calc_J(theta, X, y)
        logger.debug(f"J: {J}")
        return J
    def gradient(theta):
        # must reshape theta to (N,1):
        theta = np.reshape(theta, (-1, 1))
        # must flatten result to (N,) array for fmin_bfgs()
        grad = np.ndarray.flatten(calc_grad(theta, X, y))
        logger.debug(f"grad: {grad}")
        return grad

    theta = fmin_bfgs(cost, initial_theta, fprime=gradient)
    theta = np.reshape(theta, (-1, 1))

    print(f"Cost at theta found by fmin_bfgs: {cost(theta)}")
    print(f"Expected cost (approx): 0.203")
    print(f"theta: {theta}")
    print(f"Expected theta (approx): [ -25.161 0.206 0.201 ]")

    ax = plot_decision_boundary(theta, X, y)
    ax.set_xlabel('Exam 1 score')
    ax.set_ylabel('Exam 2 score')
    ax.legend(labels=["Admitted", "Not Admitted", "Decision Boundary"])

    # predict and accuracies
    prob = sigmoid(np.array([1, 45, 85]).dot(theta))
    print(f"For a student with scores 45 and 85, we predict an admission probability of {prob}")
    print(f"Expected value: 0.775 +/- 0.002")

    p = predict(theta, X)

    print(f"Train accuracy: {np.mean(p == y) * 100}")
    print(f"Expected accuracy (approx): 89.0")

    # show all plots at once
    plt.show()


if __name__ == "__main__":
    main()
