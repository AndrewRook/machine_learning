import numpy as np
import scipy.optimize as sciopt
from matplotlib import pyplot as plt
from scipy import stats, interpolate
from astroML.stats.random import linear
from astroML.plotting.mcmc import convert_to_stdev


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

def logL_gaussian(xi, yi, a, b):
    """gaussian log-likelihood (Eq. 5.87)"""
    xi = xi.ravel()
    yi = yi.ravel()
    a = a.reshape(a.shape + (1,))
    b = b.reshape(b.shape + (1,))
    yyi = a * xi + b
    return -0.5 * np.sum(np.log(yyi) + (yi - yyi) ** 2 / yyi, -1)


def logL_poisson(xi, yi, a, b):
    """poisson log-likelihood (Eq. 5.88)"""
    xi = xi.ravel()
    yi = yi.ravel()
    a = a.reshape(a.shape + (1,))#This is slightly faster than a[:,None], also guarantees new index will be added to the end of the array. Both are neat though!
    b = b.reshape(b.shape + (1,))
    yyi = a * xi + b

    return np.sum(yi * np.log(yyi) - yyi, -1)

def compute_5_15():
    # Original Author: Jake VanderPlas
    # Modified by Andrew Schechtman-Rook
    # License: BSD
    #   This code is based on code used to make a figure in the textbook
    #   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
    #   For more information, see http://astroML.github.com
    #   To report a bug or issue, use the following forum:
    #    https://groups.google.com/forum/#!forum/astroml-general
    #------------------------------------------------------------
    # Draw points from distribution
    np.random.seed(0)

    N = 1000

    a_true = 0.01
    xmin = 0.0
    xmax = 10.0
    b_true = 1. / (xmax - xmin) - 0.5 * a_true * (xmax + xmin)

    lin_dist = linear(xmin, xmax, a_true)
    data = lin_dist.rvs(N)

    #------------------------------------------------------------
    # Compute and plot the results
    fig = plt.figure(figsize=(9, 9))
    fig.subplots_adjust(left=0.1, right=0.95, wspace=0.3,
                        bottom=0.1, top=0.95, hspace=0.2)

    a = np.linspace(0.00001, 0.04, 71)
    b = np.linspace(0.00001, 0.15, 71)

    for num, nbins in enumerate([5, 100]):
        # divide points into bins
        yi, bins = np.histogram(data, bins=np.linspace(xmin, xmax, nbins + 1))
        xi = 0.5 * (bins[:-1] + bins[1:])

        # compute likelihoods for Poisson and Gaussian models
        factor = N * (xmax - xmin) * 1. / nbins
        #LP = logL_poisson(xi, yi, factor * a, factor * b[:, None])
        #LG = logL_gaussian(xi, yi, factor * a, factor * b[:, None])#Below is faster, marginally, probably also better practice
        LP = logL_poisson(xi, yi, factor * a, factor * b.reshape(b.shape + (1,)))
        LG = logL_gaussian(xi, yi, factor * a, factor * b.reshape(b.shape + (1,)))

        LP -= np.max(LP)
        LG -= np.max(LG)

        # find maximum likelihood point
        i, j = np.where(LP == np.max(LP))
        aP, bP = a[j[0]], b[i[0]]

        i, j = np.where(LG == np.max(LG))
        aG, bG = a[j[0]], b[i[0]]

        # plot scatter and lines
        ax = fig.add_subplot(2, 2, 1 + 2 * num)
        plt.scatter(xi, yi, s=9, c='gray', lw=0)

        x = np.linspace(xmin - 1, xmax + 1, 1000)
        for (ai, bi, s) in [(a_true, b_true, '-k'),
                            (aP, bP, '--k'),
                            (aG, bG, '-.k')]:
            px = ai * x + bi
            px[x < xmin] = 0
            px[x > xmax] = 0
            ax.plot(x, factor * px, s)

        ax.set_xlim(xmin - 1, xmax + 1)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y_i$')

        ax.text(0.04, 0.96,
                r'$\rm %i\ points$' % N + '\n' + r'$\rm %i\ bins$' % nbins,
                ha='left', va='top', transform=ax.transAxes)

        # plot likelihood contours
        ax = fig.add_subplot(2, 2, 2 + 2 * num)

        ax.contour(a, b, convert_to_stdev(LP),
                   levels=(0.683, 0.955, 0.997),
                   colors='k', linewidths=2)

        ax.contour(a, b, convert_to_stdev(LG),
                   levels=(0.683, 0.955, 0.997),
                   colors='gray', linewidths=1, linestyle='dashed')

        # trick the legend command
        ax.plot([0], [0], '-k', lw=2, label='Poisson Likelihood')
        ax.plot([0], [0], '-', c='gray', lw=1, label='Gaussian Likelihood')
        ax.legend(loc=1)

        # plot horizontal and vertical lines
        #  in newer matplotlib versions, use ax.vlines() and ax.hlines()
        ax.plot([a_true, a_true], [0, 0.2], ':k', lw=1)
        ax.plot([0, 0.06], [b_true, b_true], ':k', lw=1)

        ax.set_xlabel(r'$a^\ast$')
        ax.set_ylabel(r'$b^\ast$')

        ax.set_xlim(0, 0.04)
        ax.set_ylim(0.001, 0.15)

        ax.xaxis.set_major_locator(plt.MultipleLocator(0.02))

    fig.savefig('chap_5_5-15.png',dpi=300)



# def logl_poisson(xi_inp,yi_inp,a,b,ydeltafactor=1):
#     xi = xi_inp.ravel()
#     yi = yi_inp.ravel()
#     a = ydeltafactor*a.reshape(a.shape + (1,))
#     b = ydeltafactor*b.reshape(b.shape + (1,))
#     yyi = a * xi + b

#     #Sum over bin:
#     return np.sum(yi * np.log(yyi) - yyi, -1)

# def compute_5_15():
#     N = 50
#     a_true = 0.01
#     xmin = 0.
#     xmax = 10.
#     nbins = 5
    
#     W = xmax-xmin
#     xonehalf = (xmin+xmax)/2.
#     b_true = 1./W-a_true*xonehalf
#     lin_dist = astroML.stats.random.linear(xmin,xmax,a_true)
#     rand_data = lin_dist.rvs(N)
#     a = np.linspace(0.00001,0.04,71)
#     b = np.linspace(0.00001,0.15,71)

#     #Compute the histogram:
#     yi,bins = np.histogram(rand_data,bins=np.linspace(xmin,xmax,nbins+1))
#     xi = (bins[:-1]+bins[1:])/2.
#     logl_p = logl_poisson(xi,yi,a,b,ydeltafactor=N*(xmax-xmin)/float(nbins))
#     #Normalize:
#     logl_p -= np.max(logl_p)
#     print np.where(logl_p == np.max(logl_p))
    
if __name__ == "__main__":
    #max_entropy_dice_priors(nsides=6,mu=3.5)
    #max_entropy_dice_priors(nsides=6,mu=5.9)

    compute_5_15()
