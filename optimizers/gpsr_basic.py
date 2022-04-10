import numpy as np
import time


def GPSR_Basic(y: np.ndarray, A: np.ndarray, tau: np.ndarray, true_x: np.ndarray = None):

    def AT_(x: np.ndarray):
        return A.T@x

    def A_(x: np.ndarray):
        return A@x

    if true_x is not None:
        compute_mse = True
    else:
        compute_mse = False
    
    # Set the defaults for the optional parameters
    stopCriterion = 3 
    tolA = 0.01 
    tolD = 0.0001 
    debias = 0
    maxiter = 10000 
    maxiter_debias = 500 
    miniter = 5 
    miniter_debias = 0 
    init = 0
    verbose = True
    continuation = 1
    cont_steps = 5 
    firstTauFactorGiven = 0
    firstTauFactor = 0.5*np.max(np.abs(AT_(y)))

    realmin = np.finfo(float).tiny
    eps = np.finfo(float).eps
    # sufficient decrease parameter for GP line search
    mu = 0.1 
    # backtracking parameter for line search 
    lambda_backtrack = 0.5 
    
    # Set the defaults for outputs that may not be computed
    debias_start = 0 
    x_debias = [] 
    mses = []
    objective = []
    times_list = []
    lambdas = []

    
    if stopCriterion in [0, 1, 2, 3, 4]:
        print("error", "Unknown stopping criterion")

    # Precompute A.T@y since it'll be used a lot
    Aty = AT_(y)

    # Initialization
    if init == 0:   # initialize at zero, using AT to find the np.shape of x
        x = AT_(np.zeros_like(y))
    elif init == 1:   # initialize randomly, using AT to find the np.shape of x
        x = np.random.randn(np.shape(AT_(np.zeros_like(y)))[0], np.shape(AT_(np.zeros_like(y)))[1])
    else:   # initialize x0 = A.T@y
        x = Aty

    # now check if tau is an array  if it is, it has to
    # have the same np.shape as x
    if np.prod(np.shape(tau)) > 1:
        try:
            dummy = x*tau
        except:
            raise NameError('Parameter tau has wrong dimensions  it should be scalar or np.shape(x)')


    # if the true x was given, check its np.shape
    if compute_mse and (np.shape(true_x) != np.shape(x)):
        raise NameError('Initial x has incompatible np.shape')

    # if tau is scalar, we check its value  if it's large enough,
    # the optimal solution is the zero vector
    if np.prod(np.shape(tau)) == 1:
        aux = AT_(y)
        max_tau = np.max(np.abs(aux))
        if tau >= max_tau:
            x = np.zeros_like(aux)
            if debias:
                x_debias = x
            objective.append(0.5*(y.T@y))
            times_list.append(0)
            if compute_mse:
                mses.append(np.sum(np.power(x - true_x, 2)))
            return

    # initialize u and v
    u = x*(x >= 0)
    v = -x*(x < 0)

    # define the indicator vector or matrix of nonzeros in x
    nz_x = (x != 0.0)
    num_nz_x = np.sum(nz_x)

    # Compute and store initial value of the objective function
    resid =  y - A_(x)
    f = 0.5*(resid.T@resid) + np.sum(tau*u) + np.sum(tau*v)

    # auxiliary vector on ones, same np.shape as x
    onev = np.ones_like(x)

    # start the clock
    t0 = time.time()

    # store given tau, because we're going to change it in the
    # continuation procedure
    final_tau = tau

    # store given stopping criterion and threshold, because we're going
    # to change them in the continuation procedure
    final_stopCriterion = stopCriterion
    final_tolA = tolA

    # set continuation factors
    if continuation and (cont_steps > 1):
        # If tau is scalar, first check top see if the first factor is
        # too large (i.e., large enough to make the first
        # solution all zeros). If so, make it a little smaller than that.
        # Also set to that value as default
        if np.prod(np.shape(tau)) == 1:
            if (firstTauFactorGiven == 0) or (firstTauFactor*tau >= max_tau):
                firstTauFactor = 0.8*max_tau / tau
                print('parameter FirstTauFactor too large  changing')

        cont_factors = 10**np.arange(np.log10(firstTauFactor), 0, np.log10(1/firstTauFactor)/(cont_steps-1))
    else:
        cont_factors = np.array([1])
        cont_steps = 1


    iter_ = 1
    if compute_mse:
        mses.append(np.sum(np.power(x - true_x, 2)))


    # loop for continuation
    for cont_loop in range(cont_steps):

        tau = final_tau * cont_factors[cont_loop-1]

        if verbose:
            print(f'\nSetting tau = {tau}\n')

        if cont_loop == cont_steps:
           stopCriterion = final_stopCriterion
           tolA = final_tolA
        else:
           stopCriterion = 3
           tolA = 1e-3


        # Compute and store initial value of the objective function
        resid = y - A_(x)
        f = 0.5*(resid.T@resid) + np.sum(tau*u) + np.sum(tau*v)

        objective.append(f)
        times_list.append(time.time() - t0)

        # Compute the useful quantity resid_base
        resid_base = y - resid

        # control variable for the outer loop and iteration counter
        # cont_outer = (np.linalg.norm(projected_gradient) > 1.e-5)

        keep_going = 1

        if verbose:
            print('\nInitial obj=%10.6e, nonzeros=%7d\n', f, num_nz_x)

        while keep_going:
            x_previous = x

            # compute gradient
            temp = AT_(resid_base)
            term  =  temp - Aty
            gradu =  term + tau
            gradv = -term + tau

            # set search direction
            # du = -gradu  dv = -gradv  dx = du-dv
            dx = gradv-gradu
            old_u = u
            old_v = v

            # calculate useful matrix-vector product involving dx
            auv = A_(dx)
            dGd = auv.T@auv
            
            # calculate unconstrained minimizer along this direction, use this
            # as the first guess of steplength parameter lambda
            #  lambda0 = - (gradu.T@du + gradv.T@dv) / dGd
            
            # use instead a first guess based on the "conditional" direction
            condgradu = ((old_u > 0) + (gradu < 0)) * gradu
            condgradv = ((old_v > 0) + (gradv < 0)) * gradv
            auv_cond = A_(condgradu-condgradv)
            dGd_cond = auv_cond.T@auv_cond
            lambda0 = (gradu.T@condgradu + gradv.T@condgradv) / (dGd_cond + realmin)
            
            # loop to determine steplength, starting wit the initial guess above.
            lambda_ = lambda0
            while True:
                # calculate step for this lambda_ and candidate point
                du = np.maximum(u-lambda_*gradu, np.zeros(1)) - u
                u_new = u + du
                dv = np.maximum(v-lambda_*gradv, np.zeros(1)) - v
                v_new = v + dv
                dx = du-dv
                x_new = x + dx
                
                # evaluate function at the candidate point
                resid_base = A_(x_new)
                resid = y - resid_base
                f_new = 0.5*(resid.T@resid) + np.sum(tau*u_new) + np.sum(tau*v_new)
                # test sufficient decrease condition
                if f_new <= f + mu * (gradu.T@du + gradv.T@dv):
                    break
                lambda_ = lambda_ * lambda_backtrack
                print(f'\n reducing lambda_ to {lambda_}\n')

            u = u_new
            v = v_new
            prev_f = f
            f = f_new
            uvmin = np.minimum(u,v)
            u = u - uvmin
            v = v - uvmin
            x = u-v

            # calculate nonzero pattern and number of nonzeros (do this *always*)
            nz_x_prev = nz_x
            nz_x = (x!=0.0)
            num_nz_x = np.sum(nz_x)

            iter_ = iter_ + 1
            objective.append(f)
            times_list.append(time.time()-t0)
            lambdas.append(lambda_)

            if compute_mse:
                err = true_x - x
                mses.append(err.T@err)

            # print out stuff
            if verbose:
                print(f'It ={iter_}, obj={f}, lambda={lambda_}, nz={num_nz_x}')

            if stopCriterion == 0:
                # compute the stopping criterion based on the change
                # of the number of non-zero components of the estimate
                num_changes_active = (np.sum(nz_x!=nz_x_prev))
                if num_nz_x >= 1:
                    criterionActiveSet = num_changes_active
                else:
                    criterionActiveSet = tolA / 2
                keep_going = (criterionActiveSet > tolA)
                if verbose:
                    print(f'Delta n-zeros = {criterionActiveSet} (target = {tolA})\n')

            elif stopCriterion == 1:
                # compute the stopping criterion based on the relative
                # variation of the objective function.
                criterionObjective = np.abs(f-prev_f)/(prev_f)
                keep_going = (criterionObjective > tolA)
                if verbose:
                    print(f'Delta obj. = {criterionObjective} (target = {tolA})\n')

            elif stopCriterion == 2:
                # stopping criterion based on relative np.linalg.norm of step taken
                delta_x_criterion = np.linalg.norm(dx)/np.linalg.norm(x)
                keep_going = (delta_x_criterion > tolA)
                if verbose:
                    print(f'norm(delta x)/norm(x) = {delta_x_criterion} (target = {tolA})\n')

            elif stopCriterion == 3:
                # compute the "LCP" stopping criterion - again based on the previous
                # iterate. Make it "relative" to the np.linalg.norm of x.
                w = np.hstack((np.minimum(gradu, old_u), np.minimum(gradv, old_v)))
                criterionLCP = np.linalg.norm(w, np.Inf)
                criterionLCP = criterionLCP / np.max([1.0e-6, np.linalg.norm(old_u,np.Inf), np.linalg.norm(old_v,np.Inf)])
                keep_going = (criterionLCP > tolA)
                if verbose:
                    print(f'LCP = {criterionLCP} (target = {tolA})\n')

            elif stopCriterion == 4:
                # continue if not yeat reached target value tolA
                keep_going = (f > tolA)
                if verbose:
                    print(f'Objective = {f} (target = {tolA})\n')

            else:
                raise NameError('Unknwon stopping criterion')
            # end of the stopping criteria switch

            # take no less than miniter...
            if iter_<=miniter:
                keep_going = 1
            else: # and no more than maxiter iterations
                if iter_ > maxiter:
                    keep_going = 0
        # end of the main loop of the GP algorithm

    # end of the continuation loop

    # Print results
    if verbose:
        print('\nFinished the main algorithm!\nResults:\n')
        print(f'||A x - y ||_2^2 = {resid.T@resid}\n')
        print(f'||x||_1 = {np.sum(np.abs(x))}\n')
        print(f'Objective function = {f}\n')
        nz_x = (x!=0.0)
        num_nz_x = np.sum(nz_x)
        print(f'Number of non-zero components = {num_nz_x}\n')
        print(f'CPU time so far = {times_list[iter_-1]}\n')
        print('\n')


    # If the 'Debias' option is set to 1, we try to remove the bias from the l1
    # penalty, by applying CG to the least-squares problem obtained by omitting
    # the l1 term and fixing the zero coefficients at zero.

    # do this only if the reduced linear least-squares problem is
    # overdetermined, otherwise we are certainly applying CG to a problem with a
    # singular Hessian

    if (debias and (np.sum(x!=0)!=0)):
        if (num_nz_x > np.size(y)):
            if verbose:
                print('\n')
                print('Debiasing requested, but not performed\n')
                print('There are too many nonzeros in x\n\n')
                print('nonzeros in x: %8d, length of y: %8d\n',num_nz_x, np.size(y))

        elif (num_nz_x==0):
            if verbose:
                print('\n')
                print('Debiasing requested, but not performed\n')
                print('x has no nonzeros\n\n')

        else:
            if verbose:
              print('\n')
              print('Starting the debiasing phase...\n\n')

            x_debias = x
            zeroind = (x_debias!=0)
            cont_debias_cg = 1
            debias_start = iter_

            # calculate initial residual
            resid = A_(x_debias)
            resid = resid-y
            resid_prev = eps*np.ones_like(resid)

            rvec = AT_(resid)

            # mask out the zeros
            rvec = rvec * zeroind
            rTr_cg = rvec.T@rvec

            # set convergence threshold for the residual || RW x_debias - y ||_2
            tol_debias = tolD * (rvec.T@rvec)

            # initialize pvec
            pvec = -rvec

            # main loop
            while cont_debias_cg:

                # calculate A*p = Wt * Rt * R * W * pvec
                RWpvec = A_(pvec)
                Apvec = AT_(RWpvec)
                
                # mask out the zero terms
                Apvec = Apvec * zeroind
                
                # calculate alpha for CG
                alpha_cg = rTr_cg / (pvec.T@ Apvec)
                
                # take the step
                x_debias = x_debias + alpha_cg * pvec
                resid = resid + alpha_cg * RWpvec
                rvec  = rvec  + alpha_cg * Apvec
                
                rTr_cg_plus = rvec.T@rvec
                beta_cg = rTr_cg_plus / rTr_cg
                pvec = -rvec + beta_cg * pvec
                
                rTr_cg = rTr_cg_plus
                
                iter_ = iter_+1
                
                objective.append(0.5*(resid.T@resid) + np.sum(tau*np.abs(x_debias)))
                times_list.append(time.time() - t0)
                if compute_mse:
                    err = true_x - x_debias
                    mses.append((err.T@err))
                if verbose:
                    # in the debiasing CG phase, always use convergence criterion
                    # based on the residual (this is standard for CG)
                    print(' Iter = %5d, debias resid = %13.8e, convergence = %8.3e\n', iter_, resid.T@resid, rTr_cg / tol_debias)
    
                cont_debias_cg = (iter_-debias_start <= miniter_debias )or ((rTr_cg > tol_debias) and (iter_-debias_start <= maxiter_debias))

            if verbose:
                print('\nFinished the debiasing phase!\nResults:\n')
                print(f'||A x - y ||^2_2 = {resid.T@resid}\n')
                print(f'||x||_1 = {np.sum(np.abs(x))}\n')
                print(f'Objective function = {f}\n')
                nz = (x_debias != 0.0)
                print(f'Number of non-zero components = {np.sum(nz)}\n')
                print(f'CPU time so far = {times_list[iter_-1]}\n', )
                print('\n') 

    # mses = np.array(mses)
    if compute_mse:
        for k in range(len(mses)):
            mses[k] = mses[k]/np.size(true_x)
    
    return x, [x_debias, objective, times_list, debias_start, mses]