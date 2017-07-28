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

import sys
sys.path.append("..")
from ex3.ex3 import display_data, sigmoid


logger = logging.getLogger(__name__)


def forward_propagate(X, Theta1, Theta2):
    m = X.shape[0]
    A1 = np.c_[np.ones((m, 1)), X]
    Z2 = A1.dot(Theta1.T)
    A2 = np.c_[np.ones((m, 1)), sigmoid(Z2)]
    Z3 = A2.dot(Theta2.T)
    A3 = sigmoid(Z3)
    return A1, Z2, A2, Z3, A3


def nn_cost_function(nn_params,
                     input_layer_size,
                     hidden_layer_size,
                     num_labels,
                     X, y, lambda_):
    """Implements the neural network cost function for a two-layer
    neural network which performs classification.

    Computes the cost and gradient of the neural network. The parameters
    for the neural network are "unrolled" into the vector nn_params and
    need to be converted back into the weight matrices.

    Returns J (cost) and grad (gradient). The returned parameter `grad`
    should be an "unrolled" vector of the partial derivatives of the
    neural network.
    """

    # reshape nn_params back into Theta1 and Theta2
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)] \
        .reshape((hidden_layer_size, input_layer_size + 1))
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):] \
        .reshape((num_labels, hidden_layer_size + 1))

    m = X.shape[0]

    # create vectors from y labels
    y_matrix = (np.arange(1, num_labels + 1) == y).astype(int)

    # use feedforward to calculate the cost J
    A1, Z2, A2, Z3, h = forward_propagate(X, Theta1, Theta2)

    J = -(np.trace(y_matrix.T.dot(np.log(h)))
          + np.trace((1 - y_matrix).T.dot(np.log(1 - h)))) / m

    # regularized cost (ignore first column)
    R1 = np.sum(np.sum(np.square(Theta1[:, 1:])))
    R2 = np.sum(np.sum(np.square(Theta2[:, 1:])))
    J += lambda_ / (2 * m) * (R1 + R2)

    # Back propagation - determine partial derivatives of parameters
    D3 = (h - y_matrix).T
    D2 = Theta2[:, 1:].T.dot(D3) * sigmoid_gradient(Z2.T)
    Delta1 = D2.dot(A1)
    Delta2 = D3.dot(A2)

    # regularization of gradient (skip first column)
    R1_grad = np.c_[np.zeros((Theta1.shape[0], 1)), Theta1[:, 1:]]
    R2_grad = np.c_[np.zeros((Theta2.shape[0], 1)), Theta2[:, 1:]]

    Theta1_grad = 1.0 / m * (Delta1 + lambda_ * R1_grad)
    Theta2_grad = 1.0 / m * (Delta2 + lambda_ * R2_grad)

    grad = np.vstack([Theta1_grad.ravel()[:, None],
                      Theta2_grad.ravel()[:, None]])

    return J, grad


def sigmoid_gradient(z):
    """Returns the gradient of the sigmoid function.

    Computes the gradient of the sigmoid function evaluated at z.
    In particular, if z is a matrix or a vector, it returns the
    gradient for each element.
    """
    s = sigmoid(z)
    g = s * (1 - s)
    return g


def rand_initialize_weights(L_in, L_out):
    """Randomly initialize the weights of a layer with L_in
    incoming connections and L_out outgoing connections.
    Note: the resulting matrix is of size (L_out, 1 + L_in) as
    the first column handles the "bias" terms.
    """

    epsilon_init = 0.12  # sqrt(6) / sqrt(L_in + L_out)
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    return W


def debug_initialize_weights(fan_out, fan_in):
    """Initialize the weights of a layer with fan_in incoming connections
    and fan_out outgoing connections using a fixed strategy (for repeatibility).
    """
    W = np.zeros((fan_out, 1 + fan_in))
    W2 = np.reshape(np.sin(np.arange(1, W.size + 1)), W.shape) / 10
    return W2


def compute_numerical_gradient(J_func, theta):
    """Computes the gradient using "finite differences" and
    gives us a numerical estimate of the gradient.

    Computes the numerical gradient of J_func around theta and returns
    the numerical gradient. Returns numgrad where numgrad[i] is a
    numerical approximation of the partial derivative of J_func with
    respect to the i-th argument, evaluated at theta. I.e. numgrad[i]
    is the approximate partial derivative of J_func with respect to
    theta[i].
    """

    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4

    for p in range(theta.size):
        # set pertubation vector
        perturb[p] = e
        loss1, _ = J_func(theta - perturb)
        loss2, _ = J_func(theta + perturb)

        # compute numerical gradient
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0

    return numgrad


def check_nn_gradients(lambda_=0):
    """Creates a small neural network to check the backpropagation gradients.

    Outputs the analytical gradients produced by backprop and the computed
    numerical gradients. These two gradient computations should result in
    very similar values."""

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # generate some random test data
    Theta1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
    Theta2 = debug_initialize_weights(num_labels, hidden_layer_size)

    # reuse debug_initialize_weights to generate X
    X = debug_initialize_weights(m, input_layer_size - 1)
    y = 1 + np.mod(np.arange(1, m + 1), num_labels).T[:,None]

    nn_params = np.vstack([Theta1.ravel()[:, None],
                           Theta2.ravel()[:, None]])

    # closure:
    def cost_func(nn_params_):
        return nn_cost_function(nn_params_,
                                input_layer_size, hidden_layer_size,
                                num_labels, X, y, lambda_)

    cost, grad = cost_func(nn_params)
    numgrad = compute_numerical_gradient(cost_func, nn_params)

    print(np.c_[numgrad, grad])
    print("The above two columns should be very similar."
          "(Left is analytical gradient, right is numerical gradient")

    diff = np.linalg.norm(numgrad - grad, ord=2) / np.linalg.norm(numgrad + grad, ord=2)
    print(f"Relative difference: {diff}")


def predict(Theta1, Theta2, X):
    """Predict the label of an input given a trained neural network."""
    *_, h = forward_propagate(X, Theta1, Theta2)
    h_argmax = np.argmax(h, axis=1) + 1
    return h_argmax.T[:, None]


def main():
    logging.basicConfig(level=logging.INFO)

    input_layer_size = 400  # 20x20 input images of digits
    hidden_layer_size = 25  # 25 hidden units
    num_labels = 10         # digits 0-9 (0 is mapped to 10)

    data = scipy.io.loadmat('data/ex4data1.mat')
    X = data["X"]
    y = data["y"]
    m = X.shape[0]

    # randomly select 100 data points to display
    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[:100], :]
    display_data(sel)

    # load pre-calculated weights
    weights = scipy.io.loadmat('data/ex4weights.mat')
    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']

    # unroll parameters - the [:, None] adds an axis
    nn_params = np.vstack([Theta1.ravel()[:, None],
                           Theta2.ravel()[:, None]])

    # Part 3: Compute cost via feedforward:
    lambda_ = 0
    J, grad = nn_cost_function(nn_params, input_layer_size, hidden_layer_size,
                               num_labels, X, y, lambda_)
    print(f"Cost at parameters (loaded from ex4weights): {J}\n(this value should be about 0.287629)")

    # Part 4: Implement regularization
    lambda_ = 1
    J, grad = nn_cost_function(nn_params, input_layer_size, hidden_layer_size,
                               num_labels, X, y, lambda_)
    print(f"Cost at parameters (loaded from ex4weights): {J}\n(this value should be about 0.383770)")

    # Part 5: Sigmoid gradient
    g = sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
    print(f"Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:")
    print(g)

    # Part 6: Initializing parameters
    initial_Theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
    initial_Theta2 = rand_initialize_weights(hidden_layer_size, num_labels)

    initial_nn_params = np.vstack([initial_Theta1.ravel()[:, None],
                                   initial_Theta2.ravel()[:, None]])

    # Part 7: Implement back-propagation
    check_nn_gradients()

    # Part 8: Implement regularization
    lambda_ = 3
    check_nn_gradients(3)
    debug_J, _ = nn_cost_function(nn_params, input_layer_size, hidden_layer_size,
                               num_labels, X, y, lambda_)
    print(f"Cost at (fixed) debugging parameters (w/ lambda = {lambda_}: {debug_J}")
    print("(for lambda = 3, this value should be about 0.576051)")

    # Part 9: Training NN
    lambda_ = 0.25  # 1.0

    fmin = minimize(fun=nn_cost_function,
                    x0=initial_nn_params,
                    args=(input_layer_size, hidden_layer_size, num_labels, X, y, lambda_),
                    method='TNC',
                    jac=True,  # Expect fun() to return (cost, gradient)
                    options={'maxiter': 500})
    logger.info(f"Cost: {fmin.fun}")
    logger.info(f"Iterations: {fmin.nit}")

    nn_params_trained = fmin.x

    # reshape nn_params back into Theta1 and Theta2
    Theta1 = nn_params_trained[:hidden_layer_size * (input_layer_size + 1)] \
        .reshape((hidden_layer_size, input_layer_size + 1))
    Theta2 = nn_params_trained[hidden_layer_size * (input_layer_size + 1):] \
        .reshape((num_labels, hidden_layer_size + 1))

    # visualise weights
    display_data(Theta1[:, 1:])

    # Part 10: Implement predict
    pred = predict(Theta1, Theta2, X)
    accuracy = np.mean(pred == y.astype(int)) * 100
    print(f"Training set accuracy: {accuracy}%")

    plt.show()


if __name__ == "__main__":
    main()
