import numpy as np
import pymc
import matplotlib.pyplot as plt
import scipy.special
from sklearn.neighbors import BallTree

#Fit a gaussian to a set of 1D data:
def gaussian_fit_mu(points,sigma_guess,sample=10000,burn=1000):

    #define the priors:
    mu_p = pymc.Uniform('mu',-5,5,value=0)#Uniform prior on mu
    sigma_p = sigma_guess
    tau_p = 1./sigma_p**2
    M = pymc.Normal('M',mu_p,tau_p,observed=True,value=points)
    #Make the model:
    model = dict(M=M,mu=mu_p)
    #Run the model:
    S = pymc.MCMC(model)
    S.sample(sample,burn=burn)
    #testtraces = np.array([S.trace(s)[:] for s in S.stochastics])
    mu_sample = S.trace('mu')[:]
    traces = np.vstack([S.trace('mu')[:]])
    logp = compute_logp(S,model)
    BFmed,BFsig = estimate_bayes_factor(traces, logp, r=0.05, return_list=False)
    print "Bayesian (MCMC): {0:.3f} +/- {1:.3f}".format(np.mean(mu_sample),np.std(mu_sample))
    print "Frequentist: {0:.3f} +/- {1:.3f}".format(np.mean(points),np.std(points,ddof=1)/np.sqrt(len(points)))
    print "Bayes Factor: Median = {0:.3f}, p75-p25 = {1:.3f}".format(BFmed,BFsig)

def gaussian_fit_both(points,sample=10000,burn=1000):
    #Define the priors:
    mu_p = pymc.Uniform('mu',-5,5,value=0)
    log_sigma_p = pymc.Uniform('log_sigma',-10,10,value=0)
    @pymc.deterministic
    def sigma(log_sigma_p=log_sigma_p):
        return np.exp(log_sigma_p)
    @pymc.deterministic
    def tau(sigma=sigma):
        return 1./sigma**2
    M = pymc.Normal('M',mu_p,tau,observed=True,value=points)
    model = dict(M=M,mu=mu_p,log_sigma=log_sigma_p,sigma=sigma,tau=tau)
    S = pymc.MCMC(model)
    S.sample(sample,burn=burn)
    logp = compute_logp(S,model)
    mu_sample = S.trace('mu')[:]
    sigma_sample = S.trace('sigma')[:]
    logp = compute_logp(S,model)
    traces = np.vstack([S.trace('mu')[:],S.trace('sigma')[:]])
    BFmed,BFsig = estimate_bayes_factor(traces, logp, r=0.05, return_list=False)
    print "Two parameter Bayesian (MCMC):"
    print "mu: {0:.3f} +/- {1:.3f}".format(np.mean(mu_sample),np.std(mu_sample))
    print "sigma: {0:.3f} +/- {1:.3f}".format(np.mean(sigma_sample),np.std(sigma_sample))
    print "Bayes Factor: Median = {0:.3f}, p75-p25 = {1:.3f}".format(BFmed,BFsig)

def compute_logp(S, model):
    #From astroml, computes the log(p)s for a MCMC model:
    M = pymc.MAP(model)
    traces = np.array([S.trace(s)[:] for s in S.stochastics])
    logp = np.zeros(traces.shape[1])
    #Compute the log(p) at every coordinate of the traces:
    for i in range(len(logp)):
        logp[i] = -M.func(traces[:,i])
    return logp

def estimate_bayes_factor(traces, logp, r=0.05, return_list=False):
    """From astroml, estimates the bayes factor using the local density of points"""
    D, N = traces.shape

    # compute volume of a D-dimensional sphere of radius r
    Vr = np.pi ** (0.5 * D) / scipy.special.gamma(0.5 * D + 1) * (r ** D)

    # use neighbor count within r as a density estimator
    bt = BallTree(traces.T)
    count = bt.query_radius(traces.T, r=r, count_only=True)

    BF = logp + np.log(N) + np.log(Vr) - np.log(count)

    if return_list:
        return BF
    else:
        p25, p50, p75 = np.percentile(BF, [25, 50, 75])
        return p50, 0.7413 * (p75 - p25)

if __name__ == "__main__":
    np.random.seed(1)
    N = 1000
    mu_true = 1.4
    sig_true = 1.0
    xi = np.random.normal(mu_true,sig_true,N)
    gaussian_fit_mu(xi,sig_true+0.1)
    gaussian_fit_both(xi)
