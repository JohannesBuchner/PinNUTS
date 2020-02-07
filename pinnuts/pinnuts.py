"""
PinNUTS is not No-U-Turn-Sampling. 
PinNUTS is dynamic euclidean HMC with multinomial sampling,
but DEHMCMS doesn't sound like peanuts.

Licence
---------

The MIT License (MIT)

Copyright (c) 2012 Morgan Fouesneau
Copyright (c) 2019 Johannes Buchner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import numpy as np
from numpy import log, exp, sqrt
import tqdm

__all__ = ['pinnuts']
from .nuts import leapfrog, stop_criterion, find_reasonable_epsilon

def build_tree(theta, r, grad, v, j, epsilon, f, joint0):
    """The main recursion."""
    if (j == 0):
        # Base case: Take a single leapfrog step in the direction v.
        thetaprime, rprime, gradprime, logpprime = leapfrog(theta, r, grad, v * epsilon, f)
        jointprime = logpprime - 0.5 * np.dot(rprime, rprime.T)
        # Is the simulation wildly inaccurate?
        sprime = jointprime - joint0 > -1000
        # Set the return values---minus=plus for all things here, since the
        # "tree" is of depth 0.
        thetaminus = thetaprime[:]
        thetaplus = thetaprime[:]
        rminus = rprime[:]
        rplus = rprime[:]
        gradminus = gradprime[:]
        gradplus = gradprime[:]
        logptree = jointprime - joint0
        #logptree = logpprime
        # Compute the acceptance probability.
        alphaprime = min(1., np.exp(jointprime - joint0))
        #alphaprime = min(1., np.exp(logpprime - 0.5 * np.dot(rprime, rprime.T) - joint0))
        nalphaprime = 1
    else:
        # Recursion: Implicitly build the height j-1 left and right subtrees.
        thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, sprime, alphaprime, nalphaprime, logptree = build_tree(theta, r, grad, v, j - 1, epsilon, f, joint0)
        # No need to keep going if the stopping criteria were met in the first subtree.
        if sprime:
            if v == -1:
                thetaminus, rminus, gradminus, _, _, _, thetaprime2, gradprime2, logpprime2, sprime2, alphaprime2, nalphaprime2, logptree2 = build_tree(thetaminus, rminus, gradminus, v, j - 1, epsilon, f, joint0)
            else:
                _, _, _, thetaplus, rplus, gradplus, thetaprime2, gradprime2, logpprime2, sprime2, alphaprime2, nalphaprime2, logptree2 = build_tree(thetaplus, rplus, gradplus, v, j - 1, epsilon, f, joint0)
            # Conpute total probability of this trajectory
            logptot = np.logaddexp(logptree, logptree2)
            # Choose which subtree to propagate a sample up from.
            if np.log(np.random.uniform()) < logptree2 - logptot:
                thetaprime = thetaprime2[:]
                gradprime = gradprime2[:]
                logpprime = logpprime2
            logptree = logptot
            # Update the stopping criterion.
            sprime = sprime and sprime2 and stop_criterion(thetaminus, thetaplus, rminus, rplus)
            # Update the acceptance probability statistics.
            alphaprime = alphaprime + alphaprime2
            nalphaprime = nalphaprime + nalphaprime2

    return thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, sprime, alphaprime, nalphaprime, logptree

def tree_sample(theta, logp, r0, grad, epsilon, f, joint, maxheight=np.inf):
    # initialize the tree
    # Resample u ~ uniform([0, exp(joint)]).
    # Equivalent to (log(u) - joint) ~ exponential(1).
    #logu = float(joint - np.random.exponential(1, size=1))

    thetaminus = theta
    thetaplus = theta
    rminus = r0[:]
    rplus = r0[:]
    gradminus = grad[:]
    gradplus = grad[:]
    logptree = 0

    j = 0  # initial heigth j = 0
    s = 1  # Main loop: will keep going until s == 0.

    while (s == 1 and j < maxheight):
        # Choose a direction. -1 = backwards, 1 = forwards.
        v = int(2 * (np.random.uniform() < 0.5) - 1)

        # Double the size of the tree.
        if (v == -1):
            thetaminus, rminus, gradminus, _, _, _, thetaprime, gradprime, logpprime, sprime, alpha, nalpha, logptree2 = build_tree(
                thetaminus, rminus, gradminus, v, j, epsilon, f, joint)
        else:
            _, _, _, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, sprime, alpha, nalpha, logptree2 = build_tree(
                thetaplus, rplus, gradplus, v, j, epsilon, f, joint)

        # Use Metropolis-Hastings to decide whether or not to move to a
        # point from the half-tree we just generated.
        logptot = np.logaddexp(logptree, logptree2)
        if sprime and np.log(np.random.uniform()) < logptree2 - logptot:
            logp = logpprime
            grad = gradprime[:]
            theta = thetaprime
            #print("accepting", theta, logp)
        
        logptree = logptot
        
        # Decide if it's time to stop.
        s = sprime and stop_criterion(thetaminus, thetaplus, rminus, rplus)
        # Increment depth.
        j += 1
    #print("jumping to:", theta)
    return alpha, nalpha, theta, grad, logp

def pinnuts(f, M, Madapt, theta0, delta=0.6, epsilon=None):
    """
    Implements the multinomial Euclidean Hamiltonian Monte Carlo sampler
    described in Betancourt (2016).

    Runs Madapt steps of burn-in, during which it adapts the step size
    parameter epsilon, then starts generating samples to return.

    Note the initial step size is tricky and not exactly the one from the
    initial paper.  In fact the initial step size could be given by the user in
    order to avoid potential problems

    INPUTS
    ------
    epsilon: float
        step size
        see nuts8 if you want to avoid tuning this parameter

    f: callable
        it should return the log probability and gradient evaluated at theta
        logp, grad = f(theta)

    M: int
        number of samples to generate.

    Madapt: int
        the number of steps of burn-in/how long to run the dual averaging
        algorithm to fit the step size epsilon.

    theta0: ndarray[float, ndim=1]
        initial guess of the parameters.

    KEYWORDS
    --------
    delta: float
        targeted acceptance fraction

    OUTPUTS
    -------
    samples: ndarray[float, ndim=2]
    M x D matrix of samples generated by NUTS.
    note: samples[0, :] = theta0
    """

    if len(np.shape(theta0)) > 1:
        raise ValueError('theta0 is expected to be a 1-D array')

    D = len(theta0)
    samples = np.empty((M + Madapt, D), dtype=float)
    lnprob = np.empty(M + Madapt, dtype=float)

    logp, grad = f(theta0)
    samples[0, :] = theta0
    lnprob[0] = logp

    # Choose a reasonable first epsilon by a simple heuristic.
    if epsilon is None:
        epsilon = find_reasonable_epsilon(theta0, grad, logp, f)

    # Parameters to the dual averaging algorithm.
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    mu = log(10. * epsilon)

    # Initialize dual averaging algorithm.
    epsilonbar = 1
    Hbar = 0

    for m in tqdm.trange(1, M + Madapt):
        # Resample momenta.
        r0 = np.random.normal(0, 1, D)

        #joint lnp of theta and momentum r
        joint = logp - 0.5 * np.dot(r0, r0.T)

        # if all fails, the next sample will be the previous one
        samples[m, :] = samples[m - 1, :]
        lnprob[m] = lnprob[m - 1]
        
        alpha, nalpha, thetaprime, grad, logp = tree_sample(samples[m - 1, :], lnprob[m - 1], r0, grad, epsilon, f, joint, maxheight=10)
        samples[m, :] = thetaprime[:]
        lnprob[m] = logp

        # Do adaptation of epsilon if we're still doing burn-in.
        eta = 1. / float(m + t0)
        Hbar = (1. - eta) * Hbar + eta * (delta - alpha / float(nalpha))
        if (m <= Madapt):
            epsilon = exp(mu - sqrt(m) / gamma * Hbar)
            eta = m ** -kappa
            epsilonbar = exp((1. - eta) * log(epsilonbar) + eta * log(epsilon))
        else:
            epsilon = epsilonbar
    samples = samples[Madapt:, :]
    lnprob = lnprob[Madapt:]
    return samples, lnprob, epsilon


def test_pinnuts():
    """ Example usage of pinnuts: sampling a 2d highly correlated Gaussian distribution """

    class Counter:
        def __init__(self, c=0):
            self.c = c

    c = Counter()
    def correlated_normal(theta):
        """
        Example of a target distribution that could be sampled from using NUTS.
        (Although of course you could sample from it more efficiently)
        Doesn't include the normalizing constant.
        """

        # Precision matrix with covariance [1, 1.98; 1.98, 4].
        # A = np.linalg.inv( cov )
        A = np.asarray([[50.251256, -24.874372],
                        [-24.874372, 12.562814]])

        # add the counter to count how many times this function is called
        c.c += 1

        grad = -np.dot(theta, A)
        logp = 0.5 * np.dot(grad, theta.T)
        return logp, grad

    D = 2
    M = 10000
    Madapt = 5000
    theta0 = np.random.normal(0, 1, D)
    delta = 0.2

    mean = np.zeros(2)
    cov = np.asarray([[1, 1.98],
                      [1.98, 4]])

    print('Running HMC with dual averaging and trajectory length %0.2f...' % delta)
    samples_orig, lnprob, epsilon = pinnuts(correlated_normal, M, Madapt, theta0, delta, epsilon=0.125)
    print('Done. Final epsilon = %f.' % epsilon)
    print('(M+Madapt) / Functions called: %f' % ((M+Madapt)/float(c.c)))

    samples = samples_orig[1::10, :]
    print('Percentiles')
    print (np.percentile(samples, [16, 50, 84], axis=0))
    print('Mean')
    print (np.mean(samples, axis=0))
    print('Stddev')
    print (np.std(samples, axis=0))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        import pylab as plt
    temp = np.random.multivariate_normal(mean, cov, size=500)
    plt.subplot(1,3,1)
    plt.plot(temp[:, 0], temp[:, 1], '.')
    plt.plot(samples[:, 0], samples[:, 1], 'r+')

    plt.subplot(1,3,2)
    plt.hist(samples[:,0], bins=50)
    plt.xlabel("x-samples")

    plt.subplot(1,3,3)
    plt.hist(samples[:,1], bins=50)
    plt.xlabel("y-samples")
    plt.savefig('pinnuts.pdf', bbox_inches='tight')
    plt.close()

    samples = samples_orig[:,0]
    Nautocorr = 1000
    plt.plot([np.mean((samples[:-k] - samples[:-k].mean()) * (samples[k:] + samples[k:].mean())) / np.var(samples) for k in range(1, Nautocorr)])
    plt.ylim(-0.1, 1)
    plt.xlabel("number of steps")
    plt.ylabel("autocorrelation")
    plt.savefig('pinnuts-autocorr.pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    test_pinnuts()
