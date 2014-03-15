import numpy as np
import scipy.optimize as sciopt

def max_entropy_dice_priors(nsides = 6, mu=3.5):
    """Estimates the Baysian priors for the faces of an N-sided die, using only the mean value mu and the principle of maximum entropy."""
    #First, compute lambda1:
    lambda1 = sciopt.newton(compute_lambda1,1.,args=(nsides,mu))
    #Then, use that to get lambda0:
    lambda0 = sciopt.newton(compute_lambda0,1.,args=(lambda1,nsides,mu))

    #Finally, compute the probabilities on each face:
    ivals = np.linspace(1.,nsides,nsides)
    pivals = np.exp(-1.-lambda0)*np.exp(-ivals*lambda1)/float(nsides)
    print lambda0,lambda1
    print pivals

def compute_lambda1(lambda1,nsides,mu):
    ivals = np.linspace(1.,nsides,nsides)
    expvals = np.exp(-ivals*lambda1)
    return mu - np.sum(ivals*expvals)/np.sum(expvals)
def compute_lambda0(lambda0,lambda1,nsides,mu):
    ivals = np.linspace(1.,nsides,nsides)
    expvals = np.exp(-ivals*lambda1)
    return 1 - np.sum(expvals)*np.exp(-1-lambda0)/float(nsides)


if __name__ == "__main__":
    max_entropy_dice_priors(nsides=6,mu=3.5)
    max_entropy_dice_priors(nsides=6,mu=5.9)
