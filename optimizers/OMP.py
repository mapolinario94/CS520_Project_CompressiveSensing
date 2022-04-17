import numpy as np
import matplotlib.pyplot as plt


def omp(X, y, nonneg=True, nsoltncoef=None, maxit=200, tol=1e-3):
    # by default set max number of soltncoef to half of total possible
    if nsoltncoef is None:
        nsoltncoef = int(X.shape[1] / 2)

    ############# initialize things # algorithm step 0(Input) ############
    X_transpose = X.T
    nonzerosoltncoef = []
    soltncoef = np.zeros(X.shape[1], dtype=float).reshape(-1, 1)
    residual = y
    ypred = np.zeros(y.shape, dtype=float)
    err = np.zeros(maxit, dtype=float)

    ############ main iteration # algorithm step 1 #############
    for it in range(maxit):
        # compute residual covariance vector and check threshold
        rcov = np.dot(X_transpose, residual)
        # algorithm step 3 choose i to maximize step2
        if nonneg:
            i = np.argmax(rcov)
            rc = rcov[i]
        else:
            i = np.argmax(np.abs(rcov))
            rc = np.abs(rcov[i])

        # algorithm step 2 (nonzero components)
        if i not in nonzerosoltncoef:
            nonzerosoltncoef.append(i)

        # algorithm step 2( least squares)
        soltncoefi, _, _, _ = np.linalg.lstsq(X[:, nonzerosoltncoef], y)
        soltncoef[nonzerosoltncoef] = soltncoefi  # update solution

        ######## algorithm step 3 ########
        residual = y - np.dot(X[:, nonzerosoltncoef], soltncoefi)
        ypred = y - residual

        ######### check stopping criteria # algorithm step 4 ########
        err[it] = np.linalg.norm(residual) ** 2
        if err[it] < tol:  # converged
            print('\nConverged at', it)
            break
        if len(nonzerosoltncoef) >= nsoltncoef:  # hit max soltncoefficients
            print('\nFound solution with max number of soltncoefficients.')
            break
        if it == maxit - 1:  # max iterations
            print('\nreached the max number of iterations.')
    return soltncoef, nonzerosoltncoef, err[:(it + 1)], residual, ypred