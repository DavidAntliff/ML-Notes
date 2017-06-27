#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt


def feature_normalize(X):
    """Normalizes the features in X.
    feature_normalize(X) returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    """
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))
    X_norm = X

    # each column of X is a feature
    for feature in range(X.shape[1]):
        mu[:, feature] = np.mean(X[:, feature])
        sigma[:, feature] = np.std(X[:, feature], ddof=1)  # use ddof=1 to match MATLAB
        X_norm[:, feature] = (X[:, feature] - mu[:, feature]) / sigma[:, feature]
    return X_norm, mu, sigma


def compute_cost_multi(X, y, theta):
    """
    Compute cost for linear regression with multiple variables.
    J = compute_cost_multi(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y.
    """
    h = np.dot(X, theta)
    J = np.sum(np.square(h - y)) / (2 * len(y))
    return J


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    """Performs gradient descent to learn theta.
    theta, J_history = gradient_descent_multi(x, y, theta, alpha, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha.
    """
    m = len(y)
    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):
        h = np.dot(X, theta)
        error = h - y
        gradient = alpha / m * np.dot(X.T, error)
        theta = theta - gradient
        J_history[i] = compute_cost_multi(X, y, theta)
    return theta, J_history


def with_gradient_descent():
    data = np.loadtxt('data/ex1data2.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2]
    m = len(y)

    # y must be of correct shape
    y = np.reshape(y, (-1, 1))

    # scale features and set them to zero mean
    X, mu, sigma = feature_normalize(X)

    # add intercept term to X
    X = np.c_[np.ones(m), X]
    # or: np.concatenate((np.reshape(np.ones(m), (-1, 1)), X), 1)

    # alpha values to trial
    alphas = [0.01, 0.03, 0.1, 0.3, 1.0, 1.3]
    num_iters = 400

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, alpha in enumerate(alphas):
        theta = np.zeros((3, 1))
        theta, J_history = gradient_descent_multi(X, y, theta, alpha, num_iters)
        ax.plot(J_history, '-', linewidth=1, label=f"{alpha}", antialiased=True)
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Cost J")
    ax.legend()

    print(f"Theta computed from gradient descent: {theta}")

    # use alpha = 1.3
    alpha = 1.3
    theta, J_history = gradient_descent_multi(X, y, theta, alpha, num_iters)

    # Estimate the price of a 1650 sq-ft, 3 br house
    area_norm = (1650.0 - mu[0,0]) / sigma[0,0]
    rooms_norm = (3 - mu[0,1]) / sigma[0,1]
    price = np.dot(theta.T, np.array([1.0, area_norm, rooms_norm]).T)[0]
    print(f"Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): {price}");


def normal_equation(X, y):
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)
    return theta


def with_normal_equation():
    # Solve using closed-form Normal Equations
    data = np.loadtxt('data/ex1data2.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2]
    m = len(y)

    # y must be of correct shape
    y = np.reshape(y, (-1, 1))

    # no feature normalization required

    # add intercept term to X
    X = np.c_[np.ones(m), X]

    theta = normal_equation(X, y)

    print(f"Theta computed from normal equations: {theta}")

    # Estimate the price of a 1650 sq-ft, 3 br house
    price = np.dot(theta.T, np.array([1.0, 1650.0, 3]).T)[0]
    print(f"Predicted price of a 1650 sq-ft, 3 br house (using normal equations): {price}");


def main():
    with_gradient_descent()
    with_normal_equation()
    plt.show()


if __name__ == "__main__":
    main()
