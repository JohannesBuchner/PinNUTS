PinNUTS is not No-U-Turn-Sampler
===================================

PinNUTS🥜 is an improved Hamiltonian Monte Carlo algorithm with high
acceptance rates. It benefits from avoiding trajectories from u-turning
and a multinomial acceptance scheme. It is not NUTS.

PinNUTS is dynamic euclidean HMC with multinomial trajectory sampling,
but DEHMCMTS doesn't sound like peanuts🥜🥜🥜.

Content
-------

The package mainly contains:

* `pinnuts.pinnuts` - return samples using dyHMC
* `emcee_pinnuts.PinNUTSSampler` - emcee PinNUTS sampler, a derived class from `emcee.Sampler`

This package is forked from https://github.com/mfouesneau/NUTS

A few words about NUTS and dyHMC
---------------------------------

Hamiltonian Monte Carlo or Hybrid Monte Carlo (HMC) is a 
Markov chain Monte Carlo (MCMC) algorithm that avoids the 
random walk behavior and sensitivity to correlated parameters, 
biggest weakness of many MCMC methods. Instead, it takes a series 
of steps informed by first-order gradient information.

This feature allows it to converge much more quickly to 
high-dimensional target distributions compared to simpler methods 
such as Metropolis, Gibbs sampling (and derivatives).

However, HMC's performance is highly sensitive to two user-specified 
parameters: a step size, and a desired number of steps. 
In particular, if the number of steps is too small then the algorithm 
will just exhibit random walk behavior, whereas if it is too large 
it will waste computations.

Hoffman & Gelman introduced NUTS or the No-U-Turn Sampler, an 
extension to HMC that eliminates the need to set a number of steps. 
NUTS uses a recursive algorithm to find likely candidate points that 
automatically stops when it starts to double back and retrace its steps. 
Empirically, NUTS perform at least as effciently as and sometimes more 
effciently than a well tuned standard HMC method, without requiring 
user intervention or costly tuning runs.

Moreover, Hoffman & Gelman derived a method for adapting 
the step size parameter on the fly based on primal-dual averaging. 
NUTS can thus be used with no hand-tuning at all.

In practice, the implementation still requires a number of steps, 
a burning period and a stepsize. 
However, the stepsize will be optimized during the burning period, 
and the final values of all the user-defined values will be revised by the algorithm.

More recently, the Stan team made improvements over NUTS by altering
the method of recursively selecting points in a trajectory from
slice sampling to multinomial sampling, which increases the acceptance
rate and makes distant moves more frequent. This improvement is 
implemented here.
These improvements are nicely described in Betancourt (2016).
Additionally, Stan implements a generalised U-turning criterion. 
However, this does not seem to have a strong impact.

**References**:

* http://arxiv.org/abs/1111.4246: Matthew D. Hoffman & Andrew Gelman, 2011, "_The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo"
* http://arxiv.org/abs/1701.02434: Michael Betancourt, 2016, "A conceptual introduction to Hamiltonian Monte Carlo"

Example Usage
-------------

**sampling a 2d highly correlated Gaussian distribution**
see `pinnuts.pinnuts`


* define a log-likelihood and gradient function::

    def correlated_normal(theta):
        """ Example of a target distribution that could be sampled from using NUTS.  (Doesn't include the normalizing constant.)
        Note: 
        cov = np.asarray([[1, 1.98],
                          [1.98, 4]])
        """

        #A = np.linalg.inv( cov )
        A = np.asarray([[50.251256, -24.874372],
                        [-24.874372, 12.562814]])

        grad = -np.dot(theta, A)
        logp = 0.5 * np.dot(grad, theta.T)
        return logp, grad

* set your initial conditions: number of dimensions, _number of steps, number of adaptation/burning steps, initial guess, and initial step size.::

    D = 2
    M = 5000
    Madapt = 5000
    theta0 = np.random.normal(0, 1, D)
    delta = 0.2

    mean = np.zeros(2)
    cov = np.asarray([[1, 1.98], 
                      [1.98, 4]])

* run the sampling::

    samples, lnprob, epsilon = pinnuts(correlated_normal, M, Madapt, theta0, delta)

* some statistics: expecting mean = (0, 0) and std = (1., 4.)::

    samples = samples[1::10, :]
    print('Mean: {}'.format(np.mean(samples, axis=0)))
    print('Stddev: {}'.format(np.std(samples, axis=0)))

* a quick plot::

    import pylab as plt
    temp = np.random.multivariate_normal(mean, cov, size=500)
    plt.plot(temp[:, 0], temp[:, 1], '.')
    plt.plot(samples[:, 0], samples[:, 1], 'r+')
    plt.show()

Example usage as an EMCEE sampler
---------------------------------

see `emcee_pinnuts.test_sampler`

* define a log-likelihood function::

    def lnprobfn(theta):
        return correlated_normal(theta)[0]

* define a gradient function (if not numerical estimates are made, but slower)::

    def gradfn(theta):
        return correlated_normal(theta)[1]

* set your initial conditions: number of dimensions, _number of steps, number of adaptation/burning steps, initial guess, and initial step size::

    D = 2
    M = 5000
    Madapt = 5000
    theta0 = np.random.normal(0, 1, D)
    delta = 0.2

    mean = np.zeros(2)
    cov = np.asarray([[1, 1.98],
                      [1.98, 4]])

* run the sampling::

    sampler = PinNUTSSampler(D, lnprobfn, gradfn)
    samples = sampler.run_mcmc( theta0, M, Madapt, delta )

