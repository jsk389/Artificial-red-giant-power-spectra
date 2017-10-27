#!/usr/bin/env python3

import kplr
import numpy as np
import matplotlib.pyplot as pl
from scipy.misc import factorial
from scipy.interpolate import interp1d
from scipy.special import lpmv as legendre
import glob

#from build_distributions import Dists

class Background(object):
    """
    Class to compute background model for given mission (TESS or Kepler).

    :param _freq:
        The frequency array to be used in the construction of the background.

    :param _nyq:
        Nyquist frequency corresponding to the observations.

    :param _dnu:
        Large frequency separation of the star to be simulated.

    :param _numax:
        Frequency of maximum oscillation power of the star to be simulated.

    :param _dt:
        Cadence of the observations.

    :param _kmag:
        The magnitude of the star. This functions as Kepler magnitude for
        simulations of Kepler data and I-band magnitude for simulations of
        TESS data.

    :param mission:
        The mission that the simulated data is supposed to mimic. The option
        is either Kepler or TESS.

    """

    def __init__(self, _freq, _nyq, _dnu, _numax, _dt, _kmag,
                 mission):

        # Inputs are frequency array and mission type (default: TESS)
        self.f = _freq
        self.nyq = _nyq
        self.dnu = _dnu
        self.numax = _numax
        self.mission = mission
        self.kmag = _kmag # This functions as kepmag for Kepler and
                          # I-band mag for TESS
        self.dt = _dt # cadence
        self.bw = self.f[1]-self.f[0]

    def calc_timescales(self):
        """
        Use scaling relations from Kallinger et al. 2014 to determine
        background parameters
        """
        b1 = 0.317 * self.numax ** 0.970
        b2 = 0.948 * self.numax ** 0.992
        print("B1, B2: ", b1, b2)
        return b1, b2

    def calc_amplitudes(self):
        """
        Use scaling relations from Kallinger et al. 2014 to determine
        background parameters
        """
        if self.mission == 'Kepler':
            a1 = 3382 * self.numax ** -0.609
            a2 = 3382 * self.numax ** -0.609
        elif self.mission == 'TESS':
            a1 = 0.85 * 3382 * self.numax ** -0.609
            a2 = 0.85 * 3382 * self.numax ** -0.609
        print("A1, A2: ", a1, a2)
        return a1, a2

    def harvey_profile(self, params):
        """
        Use the profile as set out in Campante et al. 2015
        """
        return ((2.0*np.sqrt(2.0)/np.pi) * ((params[0]**2.0) / \
                      params[1])) / \
                      (1.0 + (self.f/params[1])**4.0)

    def eta_sq(self):
        """
        Compute sinc^2 modulation from sampling
        """
        return np.sinc(self.f/(2.0*self.nyq))**2.0

    def shot_noise(self):
        """
        Use results from Jenkins (2010) for long cadence shot noise
        values. Converted into ppm^2uHz^-1 using equation in
        Chaplin (2011).
        """
        if self.mission == 'Kepler':
            c = 3.46*10**(0.4*(12 - self.kmag) + 8)
            sigma = 1e6 * np.sqrt(c + 7e7) / c
            #sigma /= np.sqrt(12)
            return 2.0e-6 * sigma**2.0 * self.dt

        elif self.mission == 'TESS':
            # Need to change this to input so it is compatible with python3
            shot_val = raw_input("What value of shot noise do you want to use? (Given in units of ppm^2uHz^-1)\n")
            # Need to check type of shot_val or if None etc.
            return float(shot_val)
        else:
            sys.exit('NO MISSION SELECTED!')

    def __call__(self):
        a1, a2 = self.calc_amplitudes()
        b1, b2 = self.calc_timescales()
        m = self.eta_sq() * \
               (self.harvey_profile([a1,b1]) + \
                self.harvey_profile([a2,b2]))
        print("Pgran: ", m[int(self.numax / (self.f[1]-self.f[0]))])
        shot = self.shot_noise()
        print("Shot noise: ", shot)
        return self.eta_sq() * \
               (self.harvey_profile([a1,b1]) + \
                self.harvey_profile([a2,b2])) + \
                shot

if __name__=="__main__":

    # Let's check it works for some example values!

    dnu = 17.4
    numax = (dnu / 0.276) ** (1.0 / 0.757)
    dpi = 80.0
    epsg = 0.0
    gsplit = 0.4
    R = 0.0
    q = 0.2

    Teff = 4950
    kmag = 12.44

    dt = 29.4*60.0
    T = 4.0 * 365.25 * 86400.0
    nyq = 1.0 / (2.0 * dt)
    freq = np.arange(0, nyq, 1.0 / T)*1e6

    # Set up class for scaling relations
    backg = Background(freq, nyq*1e6, numax, dt, kmag, N=1, \
                       mission='Kepler')
    pl.plot(backg.f, backg())
    pl.xlabel(r'Frequency ($\mu$Hz)')
    pl.ylabel(r'PSD (ppm$^{2}\mu$Hz$^{-1}$)')
    pl.yscale('log')
    pl.show()
