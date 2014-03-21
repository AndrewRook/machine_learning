# Original Author: Jake VanderPlas
# Modified By: Andrew Schechtman-Rook
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
#   To report a bug or issue, use the following forum:
#    https://groups.google.com/forum/#!forum/astroml-general
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib import pyplot as plt

from sklearn.decomposition import NMF
from sklearn.decomposition import FastICA
from sklearn.decomposition import RandomizedPCA

from astroML.datasets import sdss_corrected_spectra
#from astroML.decorators import pickle_results

def SDSS_PCA():
    #Based on Figure 7.4
    #Download SDSS Data:
    data = sdss_corrected_spectra.fetch_sdss_corrected_spectra()
    spectra = sdss_corrected_spectra.reconstruct_spectra(data)
    wavelengths = sdss_corrected_spectra.compute_wavelengths(data)

    print spectra.shape,wavelengths.shape
    def compute_PCA(n_components=5):
        spec_mean = spectra.mean(axis=0)
        print spec_mean.shape

        #Randomized PCA is faster (according to astroML):
        pca = RandomizedPCA(n_components-1)
        pca.fit(spectra)
        pca_components = np.vstack([spec_mean,pca.components_])

        return pca_components

    n_components = 5
    pca_components = compute_PCA(n_components)
    fig,axs = plt.subplots(n_components,sharex=True)
    for i in range(len(axs)):
        axs[i].plot(wavelengths,pca_components[i],ls='-',color='black')
        axs[i].yaxis.set_major_locator(MaxNLocator(nbins=4,steps=[1,2,4,5,10]))
        axs[i].set_ylabel(i+1)
        #axs[i].set_yscale('log')
    axs[-1].set_xlabel(r'Wavelength ($\AA$)')
    fig.subplots_adjust(hspace=0)
    fig.text(0.02, 0.5, 'Component', ha='center', va='center', rotation='vertical')
    fig.savefig('SDSS_PCA.png',dpi=300)

if __name__ == "__main__":
    SDSS_PCA()
