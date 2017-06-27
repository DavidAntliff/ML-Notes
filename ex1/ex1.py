#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required by fig.gca(projection='3d')
from matplotlib.colors import Normalize
from matplotlib import cm


def plot_data(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, 'rx', markersize=10, label="Training data")
    ax.set_ylabel("Profit in $10,000s")
    ax.set_xlabel("Population of City in 10,000s")
    return ax


def compute_cost(X, y, theta):
    """Compute cost for linear regression
    J = compute_cost(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y.
    """
    h = np.dot(X, theta)
    J = np.sum(np.square(h - y)) / (2 * len(y))
    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """Performs gradient descent to learn theta
    theta = gradient_descent(X, y, theta, alpha, num_iters) updates theta by 
    taking num_iters gradient steps with learning rate alpha
    """
    J_history = np.zeros((num_iters, 1))
    m = y.shape[0]
    for i in range(num_iters):
        h = np.dot(X, theta)
        error = h - y
        gradient = alpha / m * np.dot(X.T, error)
        theta = theta - gradient
        J_history[i] = compute_cost(X, y, theta)
    return theta, J_history


def main():
    data = np.loadtxt('data/ex1data1.txt', delimiter=',')
    X = data[:, 0]
    y = data[:, 1]
    m = len(y)
    ax = plot_data(X, y)

    # y must be of correct shape
    y = np.reshape(y, (-1, 1))

    # add a column of ones to X
    X = np.stack((np.ones(m), X), 1)
    # or: X = np.c_[np.ones(m), X]
    theta = np.zeros((2, 1))

    iterations = 1500
    alpha = 0.01

    J = compute_cost(X, y, theta)
    print(f"With theta = [ 0 ; 0 ]\nCost computed = {J}")
    print("Expected cost value (approx) 32.07")

    J = compute_cost(X, y, np.array([[-1.], [2.]]))
    print(f"With theta = [ -1 ; 2 ]\nCost computed = {J}")
    print("Expected cost value (approx) 54.24")

    theta, _ = gradient_descent(X, y, theta, alpha, iterations)
    print(f"Theta found by gradient descent: {theta}")
    print(f"Expected theta values (approx): [[ -3.6303 ], [ 1.1664]]")

    ax.plot(X[:,1], np.dot(X, theta), '-', label="Linear regression")
    ax.legend()

    predict1 = np.dot(np.array([1, 3.5]), theta)
    print(f"For population = 35,000, we predict a profit of {predict1 * 10000}")
    predict2= np.dot(np.array([1, 7]), theta)
    print(f"For population = 70,000, we predict a profit of {predict2* 10000}")

    # Visualizing J(theta_0, theta_1)
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
            J_vals[i, j] = compute_cost(X, y, t)

    # transpose J_vals to flip axes
    J_vals = J_vals.T

    # surface plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')  # requires 'from mpl_toolkits.mplot3d import Axes3D'
    color_map = cm.jet
    scalar_map = cm.ScalarMappable(norm=Normalize(vmin=np.min(J_vals), vmax=np.max(J_vals)), cmap=color_map)
    C_colored = scalar_map.to_rgba(J_vals)

    mesh_x, mesh_y = np.meshgrid(theta0_vals, theta1_vals)
    ax.plot_surface(mesh_x, mesh_y, J_vals, rstride=1, cstride=1, facecolors=C_colored, antialiased=True)
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\theta_1$")

    # contour plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\theta_1$")
    ax.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=2)

    # show all plots at once
    plt.show()


if __name__ == "__main__":
    main()
