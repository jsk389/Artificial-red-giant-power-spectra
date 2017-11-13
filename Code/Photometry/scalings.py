#!/usr/bin/env python
# -*- coding: utf-8 -*-

import kplr
import numpy as np
import matplotlib.pyplot as pl
from scipy.misc import factorial
from scipy.interpolate import interp1d
from scipy.special import lpmv as legendre
import glob

#from build_distributions import Dists

class Scalings(object):

    def __init__(self, _Teff, _kmag, _dnu, _numax, _dpi, _epsg, _q, \
                      _gsplit, _R, _inc, _dt, _T, mission, H_env):
        """
        Class to compute asteroseismic parameters based on scaling relations.

        :param _Teff: The effective temperature of the star.
        :type _Teff: float

        :param _kmag: The magnitude of the star. This functions as Kepler magnitude for simulations of Kepler data and I-band magnitude for simulations of TESS data.
        :type _kmag: float

        :param _dnu: Large frequency separation of the star to be simulated.
        :type _dnu: float

        :param _numax: Frequency of maximum oscillation power of the star to be simulated.
        :type _numax: float

        :param _dpi: Period spacing of the l=1 mixed modes.
        :type _dpi: float

        :param _epsg: A phase term accounting for behaviour near the turning points of the modes.
        :type _epsg: float

        :param _q: Coupling factor.
        :type _q: float

        :param _gsplit: Rotational splitting of thhe underlying g-modes.
        :type _gsplit: float

        :param _R: Ratio of the average envelope rotation rate to the average core rotation rate.
        :type _R: float

        :param _inc: Inclination angle of the star (degrees).
        :type _inc: float

        :param _dt: Cadence of the instrument (seconds).
        :type _dt: float

        :param mission: The mission that the simulated data is supposed to mimic. The option is either Kepler or TESS.
        :type mission: str

        :param Henv: Height of the oscillation envelope.
        :type Henv: float

        """
        # Cadence for Kepler long cadence observations
        self.dt = _dt #29.4*60.0
        # Stellar effective temperature
        self.Teff = _Teff
        # Numax of star
        self.numax = _numax
        # Calculate delta nu from scaling relations -> need to add in option to
        # compute this only if not given
        self.dnu = _dnu
        # Period spacing
        self.dpi = _dpi
        # Width of oscillation envelope (FWHM)
        self.nuwidth = self.calc_nuwidth()
        # Number of observable radial orders -> 5 added just to produce more
        # radial orders, even if they can't be observed -> makes spectrum a
        # little more realistic
        self.n_env = int(self.n_env() + 5)
        print("NUMBER OF ORDERS: ", self.n_env)
        # Espilon g
        self.epsg = _epsg #np.zeros(1)
        # Coupling
        self.q = _q
        # Ratio of rotation rate of envelope to core
        self.R = _R
        # Maximum rotational splitting
        self.gsplit = _gsplit
        # Inclination angle
        self.inc = _inc
        # Length of observation
        self.T = _T
        # Selects which mission is to be used
        self.mission = mission
        # Compute Nyquist frequency from cadence
        self.nyq = 1.0 / (2.0 * self.dt)
        # Compute frequency bin-width from length of observation
        self.bw = 1.0 / self.T
        # Compute frequency array from calculate bin width and Nyquist
        self.freq = np.arange(0, self.nyq, self.bw)*1e6
        if self.mission == 'Kepler':
            # Sum of square of spatial response functions for all modes
            self.vis_tot = 3.16
            # Sum of square of spatial response function for l=1 modes
            self.vis1 = 1.54
            # Sum of square of spatial response function for l=2 modes
            self.vis2 = 0.58
            # Kepler magnitude
            self.kmag = _kmag
        elif self.mission == 'TESS':
            # Sum of square of spatial response functions for all modes
            self.vis_tot = 2.94
            # Sum of square of spatial response function for l=1 modes
            self.vis1 = 1.46
            # Sum of square of spatial response function for l=2 modes
            self.vis2 = 0.46
            # I-band magnitude
            self.Imag = _kmag

        if H_env is not None :
            self.H_env = H_env
        else:
            self.H_env = None

    def gamma_0(self):
        """
        Use scaling relation from Corsaro et al. 2012
        gamma = gamma0*exp[(Teff-5777)/T0]
        """
        gamma0 = np.random.normal(1.39, 0.1)
        T0 = np.random.normal(601, 3)
        gamma_0 = gamma0*np.exp((self.Teff-5777)/T0)
        return gamma0*np.exp((self.Teff-5777)/T0)

    def d02_scaling(self):
        """
        Use scaling relation from Corsaro et al. 2012
        d02 = 0.112 +/- 0.016
        """
        return np.random.normal(0.121, 0.003, self.n_env) * self.dnu + \
               np.random.normal(0.035, 0.012)

    def d01_scaling(self):
        """
        Use scaling relation from Corsaro et al. 2012
        d01 = dnu/2 + (0.109 +/- 0.012)
        """
        return self.dnu/2.0 + np.random.normal(0.109, 0.012, self.n_env)

    def n_env(self):
        """
        Use scaling relation from Mosser et al. 2011 for number
        of radial orders with detectable amplitudes
        """
        #print("WIDTH OF OSCILLATION ENVELOPE: ", self.nuwidth)
        return self.nuwidth/self.dnu

    def epsilon(self):
        """
        Use scaling relation from Corsaro et al. 2012 for epsilon
        """
        return 0.634 + 0.546*np.log10(self.dnu) #np.random.normal(0.601, 0.025) + \
#               np.random.normal(0.632, 0.032) * np.log10(self.dnu)

    def mass_scaling(self):
        """
        Use scaling relation for mass as fn as numax, dnu and Teff
        """
        # Solar numax
        self.numax_sol = 3150.0
        # Solar effective temperature
        self.Teff_sol = 5778.0
        # Solar delta nu
        self.dnu_sol = 134.9
        return (self.numax / self.numax_sol)**3 * (self.dnu / self.dnu_sol)**-4 * \
               (self.Teff / self.Teff_sol)**1.5

    def radius_scaling(self):
        """
        Use scaling relation for radius as fn of numax dnu and temp
        """
        return (self.numax / self.numax_sol) * (self.dnu / self.dnu_sol)**-2 * \
               (self.Teff / self.Teff_sol)**0.5

    def amax_scaling(self):
        """
        Use H_env scaling relation from Mosser et al. 2012 and convert
        to radial mode amplitude at nu_max
        """
        # Calculate the height of the envelope
        #if self.Henv == None:
            #self.H_env = (np.random.normal(2.03, 0.05)*1e7) * \
            #             self.numax ** np.random.normal(-2.38, 0.01)
        #else:
        #    pass
        print("MISSON: {0}".format(self.mission))
        print("H_ENV: ", self.H_env)
        if self.mission == 'Kepler':
            #choice = raw_input("Do you want to give amax? (y/n)\n")
            #if choice == 'y':
            #    self.amax = float(raw_input("Please state amax!\n"))
            #else:
            if self.H_env is not None:
                self.amax = np.sqrt(self.H_env * self.dnu / self.vis_tot)
            else:
                # Solar amax
                self.amax_sol = 2.53
                # T0 used in bolometric correction
                self.T0 = 5934.0
                # Kepler bolometric correction
                self.ck = (self.Teff / self.T0) ** 0.8
                # Calculate mass and radius from scaling relations
                self.mass = self.mass_scaling()
                self.radius = self.radius_scaling()
                # Calculate luminosity
                self.luminosity = (self.radius) ** 2.0 * \
                                  (self.Teff / self.Teff_sol)**4.0
                # Calculate amax
                self.amax = (self.amax_sol / self.ck)*self.luminosity**0.838 / \
                            (self.mass**1.32 * (self.Teff / self.Teff_sol))

        elif self.mission == 'TESS':
            width = self.gamma_0()
            # Add in amax prediction for TESS!!
            # Add in 0.85 factor for redder bandpass
            #self.amax = 0.85 * np.sqrt(H_env * self.dnu / self.vis_tot)
            answer = raw_input("Is this for a Kepler-56 like star? (y/n)\n")
            if answer == 'y':
                H_env = float(raw_input("Please give H_env value!\n"))
                self.amax = 0.85 * np.sqrt(H_env * self.dnu / self.vis_tot)
            elif answer == 'n':
                # Solar amax
                self.amax_sol = 2.53
                # T0 used in bolometric correction
                self.T0 = 5934.0
                # Kepler bolometric correction
                self.ck = (self.Teff / self.T0) ** 0.8
                # Calculate mass and radius from scaling relations
                self.mass = self.mass_scaling()
                self.radius = self.radius_scaling()
                # Calculate luminosity
                self.luminosity = (self.radius) ** 2.0 * \
                                  (self.Teff / self.Teff_sol)**4.0
                # Calculate amax
                self.amax = 0.85 * (self.amax_sol / self.ck)*self.luminosity**0.838 / \
                            (self.mass**1.32 * (self.Teff / self.Teff_sol))
                self.amax = np.sqrt(H_env * self.dnu / self.vis_tot)

        return self.amax

    def calc_nuwidth(self):
        """
        Calculate width of oscillation envelope from Mosser et al. 2012
        """
        return 0.66 * self.numax ** 0.88

    def a0(self):
        """
        Create an interpolation function to obtain radial mode amplitude
        at any given frequency
        """
        # Don't forget that nu_width scaling relation gives the FWHM
        # NOT width of Gaussian!
        width = self.calc_nuwidth() / (2.0*np.sqrt(2.0*np.log(2)))
        amplitudes = self.amax * np.sqrt(np.exp(-(self.freq - self.numax)**2.0 / \
                                    (2.0 * width ** 2.0)))
        f = interp1d(self.freq, amplitudes)
        return f


if __name__=="__main__":


    # Let's check it works for some example values!
    dnu = 17.4
    numax = (dnu / 0.276) ** (1.0 / 0.757)
    dpi = 80.0
    epsg = 0.0
    gsplit = 0.4
    R = 0.0
    inc = 45.0
    q = 0.2

    Teff = 4840
    kmag = 12.4

    # Kepler LC and 4-years of observations
    dt = 29.4*60.0
    T = 4.0 * 365.25 * 86400.0
    # Set up class for scaling relations
    scal_relns = Scalings(Teff, kmag, numax, dpi, epsg, q, gsplit, R, \
                  inc, dt, T, N=1, mission='Kepler')
    print(scal_relns.amax_scaling())
    scal_relns = Scalings(Teff, kmag, numax, dpi, epsg, q, gsplit, R, \
                  inc, dt, T, N=1, mission='TESS')
    print(scal_relns.amax_scaling())
