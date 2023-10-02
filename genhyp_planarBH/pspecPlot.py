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
from scipy.interpolate import griddata
from scipy import ndimage
from matplotlib.colors import LogNorm

###############################################################################
###############################################################################

def readEigs(file):
    # First line is the length of the vector of eigenvalues
    with open(file, 'r') as f:
        data = []
        N = 0
        for line in f.readlines():
            l = line.strip().split('\t')
            if len(l) == 1:
                N = int(l[0])
            else:
                data.append([float(l[0]),float(l[1])])
        # Turn into array of complex values
        out = np.ndarray((N,),dtype=complex)
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
        out = [val for x in sig for val in x]
        return np.matrix(out).reshape(N,N), [xmin, xmax, ymin, ymax, grid]


###############################################################################
###############################################################################

def main(args):
    if len(args) >= 2:
        data = readEigs(args[1])
        if len(args) == 3:
            sigma, pdata = readSigma(args[2])
        else:
            pass
    else:
        main(['0'])

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
    ax[0].set_xlim(-10,10)
    ax[0].set_ylim(-7,1)

    if len(args) == 3:
        # Pseudospectrum
        
        X,Y = np.mgrid[pdata[0]:pdata[1]:-1j*(pdata[-1]+1),
                        pdata[2]:pdata[3]:-1j*(pdata[-1]+1)]
        """
        # Interpolation
        XX,YY = np.mgrid[pdata[0]:pdata[1]:-1j*(2*pdata[-1]+1),
                        pdata[2]:pdata[3]:-1j*(2*pdata[-1]+1)]
        print(X)
        print(XX)
        X = np.linspace(pdata[0],pdata[1], int(pdata[-1])+1, endpoint=True)
        Y = np.linspace(pdata[2],pdata[3], int(pdata[-1])+1, endpoint=True)
        # Resample the data at a higher factor using cublic spline interpolation
        XX = ndimage.zoom(X,2)
        YY = ndimage.zoom(Y,2)
        ZZ = griddata((X,Y), sigma, (XX,YY), method='cubic')
        for i in range(ZZ.shape[0]):
            for j in range(ZZ.shape[1]):
                if ZZ[i,j] <= 0:
                    print(i,j,ZZ[i,j])

        if np.any(ZZ < 0 ):
            print(np.where(x < 0 for x in ZZ))
        print(XX.shape,YY.shape,ZZ.shape)
        """
        #############################
        #  Plot the pseudospectrum
        ############################# 
        # Use interpolation to produce a smoothed, higher order plot;
        # include a set of contour lines
        # at the desired powers of ten; a colourbar with log values in
        # scientific notation
        sigspan = int(np.floor(np.log10(np.max(sigma))) - np.floor(np.log10(np.min(sigma))))
        print(np.min(sigma),np.max(sigma))
        levels = [1 * 10 ** (np.floor(np.log10(np.min(sigma)))) * 10 ** i for i in range(0,sigspan + 2)]
        levels = [1 * 10 ** (np.floor(np.log10(np.min(sigma)))) * 10 **(2*i) for i in range(0,8)]
        #ZZ = griddata((X,Y), sigma, (XX,YY), method='nearest')
        CS = ax[1].contourf(X, Y, sigma, levels=levels, locator=ticker.LogLocator(), cmap=cm.viridis_r)
        #CS = ax[1].imshow(sigma.T, norm=LogNorm(vmin=np.min(sigma),vmax=np.max(sigma)), cmap=cm.viridis_r, origin='lower', extent=(pdata[0],pdata[1],pdata[2],pdata[3]), interpolation='antialiased')
        #ax[1].contour(X, Y, sigma, levels=levels, colors='white', linewidths=0.5)
        cb = fig.colorbar(CS, ax=ax[1],
                            format=ticker.LogFormatterSciNotation(base=10.0))
        cb.set_label(label=r'      $\sigma^\epsilon$', rotation='horizontal')
        ax[1].plot([e.real for e in data], [e.imag for e in data], 'rx',
                markersize=8)
        ax[1].set_xlabel(r'Re $\omega$')
        ax[1].set_ylabel(r'Im $\omega$')
        ax[1].set_xlim(pdata[0],pdata[1])
        ax[1].set_ylim(pdata[2],pdata[3])
        ax[0].set_xlim(pdata[0],pdata[1])
        ax[0].set_ylim(pdata[2],pdata[3])
    else:
        pass
    plt.savefig('data/pspec_N%d' % int(len(data)/2) + '.pdf', format='pdf',
                transparent=True, bbox_inches='tight')
    plt.show()

###############################################################################
###############################################################################

if len(sys.argv) == 1:
    print("\nPlot a given set of eigenvalues and pseudospectrum")
    print("usage: python eigenplot.py jEigenvals.txt jPspec.txt")
    print("\tjEigenvals.txt: a text file containing eigenvalues")
    print("\tjPspec.txt: a text file containing the pseudospectrum plotting area and data")
else:
    main(sys.argv)

###############################################################################
###############################################################################

