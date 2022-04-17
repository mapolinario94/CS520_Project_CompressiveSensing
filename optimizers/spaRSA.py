import numpy as np
from numpy import linalg as LA
import sys
import matplotlib.pyplot as plt

def f(A, x, y):
    residual = A @ x - y
    return residual.T @ residual / 2

def gradient(A, x, y):
    return A.T @ (A @ x - y)

def phi(A, x, y, tau, norm=1):
    return f(A, x, y) + tau * LA.norm(x, norm)

def soft(u, a):
    # u is a vector, a is a scalar
    # element-wise operations
    return np.sign(u) * np.maximum(np.abs(u) - a, 0)

def hard(u, a):
    # u is a vector, a is a scalar
    # element-wise operations
    return u * (np.abs(u) > a)

def solve_sub_prob(u, tau, alpha, norm=1):
    # u: vector
    # tau, alpha: scalar
    # norm: 1 or 0 (either l-1 or l-0 norm)
    # assume we do not use any other norms
    if norm == 1:
        return soft(u, tau / alpha)
    return hard(u, np.sqrt(2 * tau / alpha))

def barzilai_borwein(A, x, prev_x):
    alpha_min = 1e-10 # constant alpha min
    alpha_max = 1e10 # constant alpha max
    # A: given matrix
    # x: x_k
    # prev_x: x_(k-1)
    s = x - prev_x
    # add fl. pt. correction
    alpha = (A @ s).T @ (A @ s) / (sys.float_info.min + s.T @ s)
    alpha = min(alpha_max, max(alpha_min, alpha))
    return alpha


def spaRSA(A, y, tau, norm=1):
    eta = 1.5  # factor, eta > 1
    k = 0  # iter counter
    x = np.zeros(A.shape[1]).reshape(-1,1)  # init guess of x
    alpha = 1.0  # init alpha
    tol = 1e-4

    while True:
        # save current value
        prev_x = x
        prev_phi = phi(A, prev_x, y, tau, norm)
        # find the next x
        while True:
            u = prev_x - gradient(A, prev_x, y) / alpha
            x = solve_sub_prob(u, tau, alpha, norm)  # update x_k
            alpha *= eta  # update alpha_k
            curr_phi = phi(A, x, y, tau, norm)
            if curr_phi < prev_phi:
                break  # the acceptance criterion is satisfied

        k += 1  # update counter
        alpha = barzilai_borwein(A, x, prev_x)  # choose the next alpha
        if np.abs(curr_phi - prev_phi) / prev_phi < tol:
            break  # stop criterion: no more relative decrease
    # end of outer loop
    yhat = A @ x
    return x, yhat