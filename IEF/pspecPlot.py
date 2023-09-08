# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:11:41 2023

@author: bradc
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from matplotlib.gridspec import GridSpec

###############################################################################
###############################################################################

def readEigs(file):
    # First line is the length of the vector of eigenvalues
    with open(file, 'r') as f:
        data = []
        for line in f.readlines():
            l = line.strip().split('\t')
            if len(l) == 1:
                pass
            else:
                data.append([float(l[0]),float(l[1])])
        # Turn into array of complex values
        out = np.ndarray((len(data),),dtype=complex)
        for i in range(len(data)):
            out[i] = data[i][0] + data[i][1]*1j
        return out

def readSigma(file):
    with open(file, 'r') as f:
        sig = []
        xmin, xmax, ymin, ymax, grid = 0.,0.,0.,0.,0.
        # First line is the bounds of the pseudospectrum and number of
        # grid points
        for line in f.readlines():
            l = line.strip().split('\t')
            if len(l) == 5:
                xmin = float(l[0])
                xmax = float(l[1])
                ymin = float(l[2])
                ymax = float(l[3])
                grid = float(l[4])
            elif len(l) == 1:
                N = int(l[0][1:-1].split(',')[0])
            else:
                sig.append([float(val) for val in l])
        out = np.ndarray((N,N))
        for i in range(len(sig)):
            out[i,:] = sig[i]
        return out, [xmin, xmax, ymin, ymax, grid]


###############################################################################
###############################################################################

def main(args):
    print(args)
    data = readEigs(args[1])
    print(data)
    if len(args) > 2:
        sigma, pdata = readSigma(args[2])
    else:
        pass

    # use LaTeX fonts in the plot
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')


    fig = plt.figure(figsize=(10,8))
    gs = GridSpec(2,1)
    ax = []
    ax.append(fig.add_subplot(gs[0,0]))
    ax.append(fig.add_subplot(gs[1,0]))


    ax[0].plot([e.real for e in data], [e.imag for e in data], 'C0.')
    ax[0].grid(ls='--', alpha=0.5)
    ax[0].set_xlabel(r'Re $\omega$')
    ax[0].set_ylabel(r'Im $\omega$')
    #ax[0].set_xlim(-10,10)
    #ax[0].set_ylim(-7,1)

    if len(args) > 2:
        # Pseudospectrum
        [X,Y] = np.mgrid[pdata[0]:pdata[1]:-1j*(pdata[-1]+1),
                        pdata[2]:pdata[3]:-1j*(pdata[-1]+1)]


        # Plot the pseudospectrum: a filled contour plot; a set of contour lines
        # at the desired powers of ten; a colourbar with log values in
        # scientific notation
        sigspan = int(np.floor(np.log10(np.max(sigma))) - np.floor(np.log10(np.min(sigma))))
        print(np.min(sigma),np.max(sigma))
        levels = [1 * 10 ** (np.floor(np.log10(np.min(sigma)))) * 10 ** i for i in range(0,sigspan + 2)]
        #levels = [1e-5 * 10 ** (i/3) for i in range(0,16)]
        CS = ax[1].contourf(X, Y, sigma, levels=levels, locator=ticker.LogLocator(),
                        cmap=cm.viridis_r)
        
        #ax[1].contour(X, Y, sigma, levels=[1 * 10 ** (np.floor(np.log10(np.min(sigma))) + i) for i in range(0, sigspan+1)], colors='white', linewidths=0.5)
        cb = fig.colorbar(CS, ax=ax[1],
                            format=ticker.LogFormatterSciNotation(base=10.0))
        cb.set_label(label=r'      $\sigma^\epsilon$', rotation='horizontal')
        ax[1].plot([e.real for e in data], [e.imag for e in data], 'rx',
                markersize=8)
        ax[1].set_xlabel(r'Re $\omega$')
        ax[1].set_ylabel(r'Im $\omega$')
        ax[1].set_xlim(pdata[0],pdata[1])
        ax[1].set_ylim(pdata[2],pdata[3])
    else:
        pass
    #plt.savefig('data/pspec_N%d' % Nspec + '.pdf', format='pdf',
    #            transparent=True, bbox_inches='tight')
    plt.show()

###############################################################################
###############################################################################

if len(sys.argv) == 1:
    print("n\Plot a given set of eigenvalues and pseudospectrum")
    print("usage: python eigenplot.py jEigenvals.txt jPspec.txt")
    print("\tjEigenvals.txt: a text file containing eigenvalues")
    print("\tjPspec.txt: a text file containing the pseudospectrum plotting area and data")
else:
    main(sys.argv)

###############################################################################
###############################################################################

