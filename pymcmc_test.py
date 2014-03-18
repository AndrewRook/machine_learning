import numpy as np
import pymc
import matplotlib.pyplot as plt

#Fit a gaussian to a set of 1D data:
def gaussian_fit_mu(points,sigma_guess,sample=10000,burn=1000):

    #define the priors:
    mu_p = pymc.Uniform('mu',-5,5,value=0)#Uniform prior on mu
    sigma_p = sigma_guess
    tau_p = 1./sigma**2
    M = pymc.Normal('M',mu_p,tau_p,observed=True,value=points)
    #Make the model:
    model = dict(M=M,mu=mu_p)
    #Run the model:
    S = pymc.MCMC(model)
    S.sample(sample,burn=0)
    mu_fullsample = S.trace('mu')[:]
    mu_sample_postburn = mu_fullsample[burn:]
    print "Bayesian (MCMC): {0:.3f} +/- {1:.3f}".format(np.mean(mu_sample_postburn),np.std(mu_sample_postburn))
    print "Frequentist: {0:.3f} +/- {1:.3f}".format(np.mean(points),np.std(points,ddof=1)/np.sqrt(len(points)))
    ax = plt.figure().add_subplot(111)
    ax.plot(mu_fullsample[:1000],ls='-',color='black')
    ax.plot(mu_sample_postburn[:1000],ls='-',color='red')
    ax.set_xlabel("Iteration Number")
    ax.set_ylabel(r"Current \mu Estimate")
    ax.figure.savefig('pymcmc_test_sigfixed.png',dpi=300)

def gaussian_fit_both(points,sample=10000,burn=1000):
    #Define the priors:
    mu_p = pymc.Uniform('mu',-5,5)
    log_sigma_p = pymc.Uniform('log_sigma',-10,10,value=0)
    @pymc.deterministic
    def sigma_p(log_sigma_p=log_sigma_p):
        return np.exp(log_sigma_p)
    @pymc.deterministic
    def tau_p(sigma_p=sigma_p):
        return 1./sigma_p**2
    M = pymc.Normal('M',mu_p,tau_p,observed=True,value=points)
    model = dict(mu=mu_p,log_sigma=log_sigma_p,sigma=sigma_p,tau=tau_p)
    S = pymc.MCMC(model)
    S.sample(sample,burn=0)
    mu_fullsample = S.trace('mu')[:]
    sigma_fullsample = np.exp(S.trace('log_sigma')[:])
    mu_sample_postburn = mu_fullsample[burn:]
    sigma_sample_postburn = sigma_fullsample[burn:]
    print "Two parameter Bayesian (MCMC):"
    print "mu: {0:.3f} +/- {1:.3f}".format(np.mean(mu_sample_postburn),np.std(mu_sample_postburn))
    print "sigma: {0:.3f} +/- {1:.3f}".format(np.mean(sigma_sample_postburn),np.std(sigma_sample_postburn))
    
if __name__ == "__main__":
    #np.random.seed(1)
    N = 10000
    mu_true = 0.5
    sig_true = 2
    xi = np.random.normal(mu_true,sig_true,N)
    gaussian_fit_mu(xi,sig_true)
    gaussian_fit_both(xi)
