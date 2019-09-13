# Machine Learning HW1

import matplotlib.pyplot as plt
import numpy as np
# more imports

# Parse the file and return 2 numpy arrays
def load_data_set(filename):
    x = np.loadtxt(filename, usecols=(0, 1))
    y = np.loadtxt(filename, usecols=(2))
    print("x shape = ", x.shape)
    print("y shape = ", y.shape)
    return x, y

# Find theta using the normal equation
def normal_equation(x, y):
    matrix_1 = np.matmul(x.T, x)
    matrix_2 = np.linalg.inv(matrix_1)
    matrix_3 = np.matmul(matrix_2, x.T)
    theta = np.matmul(matrix_3, y)
    return theta

# Find thetas using stochiastic gradient descent
# Don't forget to shuffle
def stochiastic_gradient_descent(x, y, learning_rate, num_iterations):
    thetas = []
    theta = np.random.randn(2)

    for iteration in range(num_iterations):
        # shuffle
        idx = np.random.permutation(len(x))
        # summation
        for i in idx:
            parameter = y[i] - np.matmul(x[i], theta.T)
            theta += learning_rate * parameter * x[i]

        thetas.append(theta.copy())

    return thetas

# Find thetas using gradient descent
def gradient_descent(x, y, learning_rate, num_iterations):
    thetas = []
    theta = np.random.randn(2)

    for iteration in range(num_iterations):
        vector = y - np.matmul(x, theta.T)
        theta += learning_rate * np.matmul(x.T, vector)
        thetas.append(theta.copy())

    return thetas

# Find thetas using minibatch gradient descent
# Don't forget to shuffle
def minibatch_gradient_descent(x, y, learning_rate, num_iterations, batch_size):
    thetas = []
    theta = np.random.randn(2)

    for iteration in range(num_iterations):
        # shuffle
        idx = np.random.permutation(len(x))
        # summation
        for i in range(0, len(idx), batch_size):
            indices = idx[i:i+batch_size]
            x_batch = x[indices]
            y_batch = y[indices]
            vector = y_batch - np.matmul(x_batch, theta.T)
            theta += learning_rate * np.matmul(x_batch.T, vector)

        thetas.append(theta.copy())

    return thetas

# Given an array of x and theta predict y
def predict(x, theta):
    # theta: 1*2, x: 200*2, y: 200*1
    # print("predict, theta = ", theta)
    y_predict = np.matmul(x, theta.T)
    return y_predict

# Given an array of y and y_predict return loss
def get_loss(y, y_predict):
    loss = (np.square(y_predict-y)).mean()
    return loss

# Given a list of thetas one per epoch
# this creates a plot of epoch vs training error
def plot_training_errors(x, y, thetas, title):
    losses = []
    epochs = []
    losses = []
    epoch_num = 1
    for theta in thetas:
        losses.append(get_loss(y, predict(x, theta)))
        epochs.append(epoch_num)
        epoch_num += 1
    plt.plot(epochs, losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.show()

# Given x, y, y_predict and title,
# this creates a plot
def plot(x, y, theta, title):
    # plot
    y_predict = predict(x, theta)
    plt.scatter(x[:, 1], y)
    plt.plot(x[:, 1], y_predict)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    x, y = load_data_set('regression-data.txt')
    # plot
    plt.scatter(x[:, 1], y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Scatter Plot of Data")
    plt.show()

    theta = normal_equation(x, y)
    plot(x, y, theta, "Normal Equation Best Fit")

    # You should try multiple non-zero learning rates and  multiple different (non-zero) number of iterations
    thetas = gradient_descent(x, y, 1e-3, 100)
    print("GD thetas = ", thetas)
    plot(x, y, thetas[-1], "Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Gradient Descent Mean Epoch vs Training Loss")

    # You should try multiple non-zero learning rates and  multiple different (non-zero) number of iterations
    thetas = stochiastic_gradient_descent(x, y, 1e-3, 100) # Try different learning rates and number of iterations
    print("SGD thetas = ", thetas)
    plot(x, y, thetas[-1], "Stochiastic Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Stochiastic Gradient Descent Mean Epoch vs Training Loss")

    # You should try multiple non-zero learning rates and  multiple different (non-zero) number of iterations
    thetas = minibatch_gradient_descent(x, y, 1e-3, 100, 10)
    print("Minibatch GD thetas = ", thetas)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Minibatch Gradient Descent Mean Epoch vs Training Loss")