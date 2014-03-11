import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec
from sklearn.mixture import GMM

def exp_max_test(max_wij_resid_sum=0.0001):
    # Author: Andrew Schechtman-Rook
    # This is an attempt to implement the expectation maximum algorithm from 4.4.3.
    # First I'll try to implement it for Gaussians, but I think it would also work for a sum of exponentials
    true_a = np.array([400,500,600])
    true_mu = np.array([0.5,0.0,-0.2])
    true_sig = np.array([0.1,0.3,0.5])
    xvals = np.zeros(0)
    for i in range(len(true_a)):
        xvals = np.append(xvals,np.random.normal(loc=true_mu[i],scale=true_sig[i],size=true_a[i]))

    numgaussians_guess = 3
    sig_guess = np.ones(numgaussians_guess)*np.std(xvals,ddof=1)
    a_guess = np.ones(numgaussians_guess)/float(numgaussians_guess)
    mu_guess = xvals[np.random.randint(0,len(xvals),size=numgaussians_guess)]
    wij_guess = compute_wij(a_guess,mu_guess,sig_guess,xvals)
    
    #print sig_guess
    #print a_guess
    #print mu_guess

    converged = False
    while not converged:
        new_a = compute_a(wij_guess)
        new_mu = compute_mu(wij_guess,xvals)
        new_sig = compute_sig(wij_guess,new_mu,xvals)
        new_wij = compute_wij(new_a,new_mu,new_sig,xvals)
        wij_resids = np.abs(new_wij-wij_guess)
        sum_resids = np.sum(wij_resids)
        a_guess = new_a
        mu_guess = new_mu
        sig_guess = new_sig
        wij_guess = new_wij

        if sum_resids <= max_wij_resid_sum:
            converged = True

    plot_xes = np.linspace(xvals.min(),xvals.max(),1000)
    plot_yes = plot_xes*0
    for i in range(len(a_guess)):
        plot_yes += a_guess[i]*np.exp(-(plot_xes-mu_guess[i])**2/(2*sig_guess[i]**2))/(sig_guess[i]*np.sqrt(2*math.pi))
        
    sorted_true = np.argsort(true_a)
    sorted_guess = np.argsort(a_guess)
    print true_a[sorted_true]/float(np.sum(true_a)),a_guess[sorted_guess]
    print true_mu[sorted_true],mu_guess[sorted_guess]
    print true_sig[sorted_true],sig_guess[sorted_guess]
    ax = plt.figure().add_subplot(111)
    ax.plot(plot_xes,plot_yes,ls='-',color='black',lw=3)
    ax.hist(xvals,30,normed=True,histtype='stepfilled',alpha=0.4)
    ax.figure.savefig('exp_max_test.png',dpi=300)

def compute_wij(a,mu,sig,xvals):
    gaussvals = np.zeros((len(a),len(xvals)))
    gauss_sum = np.zeros(len(xvals))
    for i in range(len(a)):
        gaussvals[i,:] = a[i]*np.exp(-(xvals-mu[i])**2/(2*sig[i]**2))/(sig[i]*np.sqrt(2*math.pi))
        gauss_sum += gaussvals[i,:]
    return gaussvals/gauss_sum
def compute_a(wij):
    return np.sum(wij,axis=1)/float(wij.shape[1])
def compute_mu(wij,xvals):
    numerator = np.sum(wij*xvals,axis=1)
    denominator = np.sum(wij,axis=1)
    return numerator/denominator
def compute_sig(wij,xvals,mu):
    stacked_xvals = np.vstack([xvals for i in range(len(mu))])
    numerator = np.sum(wij*(stacked_xvals.T-mu)**2,axis=1)
    denominator = np.sum(wij,axis=1)
    return np.sqrt(numerator/denominator)
    
    
def make_fig_4_1():
    # Author: Jake VanderPlas
    # Modified by Andrew Schechtman-Rook
    # License: BSD
    #   The figure produced by this code is published in the textbook
    #   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
    #   For more information, see http://astroML.github.com
    #   To report a bug or issue, use the following forum:
    #    https://groups.google.com/forum/#!forum/astroml-general

    #----------------------------------------------------------------------
    # This function adjusts matplotlib settings for a uniform feel in the textbook.
    # Note that with usetex=True, fonts are rendered with LaTeX.  This may
    # result in an error if LaTeX is not installed on your system.  In that case,
    # you can set usetex to False.
    from astroML.plotting import setup_text_plots
    #setup_text_plots(fontsize=8, usetex=True)

    #------------------------------------------------------------
    # Generate Dataset
    np.random.seed(1)

    N = 50
    L0 = 10
    dL = 0.2

    t = np.linspace(0, 1, N)
    L_obs = np.random.normal(L0, dL, N)

    #------------------------------------------------------------
    # Plot the results
    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(left=0.1, right=0.95, wspace=0.05,
                        bottom=0.1, top=0.95, hspace=0.05)

    y_vals = [L_obs, L_obs, L_obs, L_obs + 0.5 - t ** 2]
    y_errs = [dL, dL * 2, dL / 2, dL]
    titles = ['correct errors',
              'overestimated errors',
              'underestimated errors',
              'incorrect model']

    for i in range(4):
        ax = fig.add_subplot(2, 2, 1 + i, xticks=[])

        # compute the mean and the chi^2/dof
        mu = np.mean(y_vals[i])
        z = (y_vals[i] - mu) / y_errs[i]
        chi2 = np.sum(z ** 2)
        chi2dof = chi2 / (N - 1)

        # compute the standard deviations of chi^2/dof
        sigma = np.sqrt(2. / (N - 1))
        nsig = (chi2dof - 1) / sigma

        # plot the points with errorbars
        ax.errorbar(t, y_vals[i], y_errs[i], fmt='.k', ecolor='gray', lw=1)
        ax.plot([-0.1, 1.3], [L0, L0], ':k', lw=1)

        # Add labels and text
        ax.text(0.95, 0.95, titles[i], ha='right', va='top',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', fc='w', ec='k'))
        ax.text(0.02, 0.02, r'$\hat{\mu} = %.2f$' % mu, ha='left', va='bottom',
                transform=ax.transAxes)
        ax.text(0.98, 0.02,
                r'$\chi^2_{\rm dof} = %.2f\, (%.2g\,\sigma)$' % (chi2dof, nsig),
                ha='right', va='bottom', transform=ax.transAxes)

        # set axis limits
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(8.6, 11.4)

        # set ticks and labels
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))

        if i > 1:
            ax.set_xlabel('observations')

        if i % 2 == 0:
            ax.set_ylabel('Luminosity')
        else:
            ax.yaxis.set_major_formatter(plt.NullFormatter())

    fig.savefig('chap_4-make_fig_4_1.eps')

def make_fig_4_2():
    # Author: Jake VanderPlas
    # Modified by Andrew Schechtman-Rook

    # License: BSD
    #   The figure produced by this code is published in the textbook
    #   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
    #   For more information, see http://astroML.github.com
    #   To report a bug or issue, use the following forum:
    #    https://groups.google.com/forum/#!forum/astroml-general

    #----------------------------------------------------------------------
    # This function adjusts matplotlib settings for a uniform feel in the textbook.
    # Note that with usetex=True, fonts are rendered with LaTeX.  This may
    # result in an error if LaTeX is not installed on your system.  In that case,
    # you can set usetex to False.

    #------------------------------------------------------------
    # Set up the dataset.
    #  We'll use scikit-learn's Gaussian Mixture Model to sample
    #  data from a mixture of Gaussians.  The usual way of using
    #  this involves fitting the mixture to data: we'll see that
    #  below.  Here we'll set the internal means, covariances,
    #  and weights by-hand.
    np.random.seed(1)

    gmm = GMM(3, n_iter=1)
    gmm.means_ = np.array([[-1], [0], [3]])
    gmm.covars_ = np.array([[1.5], [1], [0.5]]) ** 2
    gmm.weights_ = np.array([0.3, 0.5, 0.2])

    X = gmm.sample(10000)

    #------------------------------------------------------------
    # Learn the best-fit GMM models
    #  Here we'll use GMM in the standard way: the fit() method
    #  uses an Expectation-Maximization approach to find the best
    #  mixture of Gaussians for the data

    # fit models with 1-10 components
    N = np.arange(1, 11)
    models = [None for i in range(len(N))]

    for i in range(len(N)):
        models[i] = GMM(N[i]).fit(X)

    # compute the AIC and the BIC
    AIC = [m.aic(X) for m in models]
    BIC = [m.bic(X) for m in models]

    #------------------------------------------------------------
    # Plot the results
    #  We'll use three panels:
    #   1) data + best-fit mixture
    #   2) AIC and BIC vs number of components
    #   3) probability that a point came from each component

    fig = plt.figure(figsize=(9,3))
    #fig.subplots_adjust(left=0.12, right=0.97, bottom=0.21, top=0.9, wspace=0.5)
    gs = matplotlib.gridspec.GridSpec(1,3)

    # plot 1: data + best-fit mixture
    ax = fig.add_subplot(gs[0,0])
    M_best = models[np.argmin(AIC)]

    x = np.linspace(-6, 6, 1000)
    logprob, responsibilities = M_best.eval(x)
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    ax.hist(X, 30, normed=True, histtype='stepfilled', alpha=0.4)
    ax.plot(x, pdf, '-k')
    ax.plot(x, pdf_individual, '--k')
    ax.text(0.04, 0.96, "Best-fit Mixture",
            ha='left', va='top', transform=ax.transAxes)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$p(x)$')


    # plot 2: AIC and BIC
    ax = fig.add_subplot(gs[0,1])
    ax.plot(N, AIC, '-k', label='AIC')
    ax.plot(N, BIC, '--k', label='BIC')
    ax.set_xlabel('n. components')
    ax.set_ylabel('information criterion')
    ax.legend(loc=2)


    # plot 3: posterior probabilities for each component
    ax = fig.add_subplot(gs[0,2])

    p = M_best.predict_proba(x)
    p = p[:, (1, 0, 2)]  # rearrange order so the plot looks better
    p = p.cumsum(1).T

    ax.fill_between(x, 0, p[0], color='gray', alpha=0.3)
    ax.fill_between(x, p[0], p[1], color='gray', alpha=0.5)
    ax.fill_between(x, p[1], 1, color='gray', alpha=0.7)
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 1)
    ax.set_xlabel('$x$')
    ax.set_ylabel(r'$p({\rm class}|x)$')

    ax.text(-5, 0.3, 'class 1', rotation='vertical')
    ax.text(0, 0.5, 'class 2', rotation='vertical')
    ax.text(3, 0.3, 'class 3', rotation='vertical')

    fig.tight_layout()
    fig.savefig('chap_4-make_fig_4_2.png',dpi=300)
    
if __name__ == '__main__':
    exp_max_test()
