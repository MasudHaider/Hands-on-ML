import numpy as np


def gradient_descent(alpha, x, y, ep, max_iter=100):
    converged = False
    iterations = 0
    m = x.shape[0]  # number of samples

    # initial intercept(w0) and slope(w1)
    w0 = 0
    w1 = 0

    # total error on whole dataset, j(theta)
    j = sum([(w0 + w1 * x[i] - y[i]) ** 2 for i in range(m)]) / (2 * m)

    # Iterate Loop
    while not converged:
        # for each training sample, compute the gradient (d/d_theta j(theta))
        grad0 = (1.0 / m) * sum([(w0 + w1 * x[i] - y[i]) for i in range(m)])
        grad1 = (1.0 / m) * sum([(w0 + w1 * x[i] - y[i]) * x[i] for i in range(m)])

        # update the theta parameters
        w0 = w0 - alpha * grad0
        w1 = w1 - alpha * grad1

        # mean squared error/new total error on whole dataset
        e = sum([(w0 + w1 * x[i] - y[i]) ** 2 for i in range(m)]) / (2 * m)

        if abs(j - e) <= ep:
            print('Converged, iterations: ', iterations)
            converged = True

        j = e  # update error
        iterations += 1  # update iteration

        if iterations == max_iter:
            print('Max iterations reached!!!')
            converged = True

    return w0, w1


if __name__ == '__main__':

    x = np.array([1, 2, 4, 3, 5])
    y = np.array([1, 3, 3, 2, 5])

    alpha_lr = 0.01  # learning rate
    ep_cr = 0.001  # convergence criteria

    # call gradient_descent, and get intercept(theta0) and slope(theta1)
    theta0, theta1 = gradient_descent(alpha_lr, x, y, ep_cr)
    print('theta0 = %s theta1 = %s' % (theta0, theta1))
