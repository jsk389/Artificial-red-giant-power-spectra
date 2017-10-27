# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from background import Background
import numpy as np
import matplotlib.pyplot as pl
from scalings import Scalings
from radial_modes import RadialModes

from scipy.misc import factorial
from scipy.interpolate import interp1d
from scipy.special import lpmv as legendre

class QuadrupoleModes(object):
    """
    Class to compute radial modes from inputs
    """
    def __init__(self, radial, dipole, model, _freqs_l2=[]):

        # Initialise radial modes class to deal with some parameters
        # Such as nominal p-mode amplitude and width
        # Don't want to do it this way!! - a better way to do this?!
        # Define number of orders
        #self.model = _model

        # calc l=2 mode frequencies
        self.radial = radial # Using radial mode class values
        self.dipole = dipole
        self.model = model
        self.d02 = self.radial.d02_scaling()
        self.d02 = np.array([self.d02]*len(self.radial.freq_0)) #self.d02[:len(self.radial.freq_0)]
        if len(_freqs_l2) == 0:
            self.freq_2 = self.radial.freq_0 - self.d02
        else:
            self.freq_2 = _freqs_l2
        self.l2_split = self.dipole.l2_split
        print("l=0 freqs: ", self.radial.freq_0)
        print("l=2 freqs: ", self.freq_2)

    def create_quadrupole_modes(self):
        """
        Create l=2 modes (assuming p-like)
        """
        mod = np.zeros(len(self.radial.f))
        self.amp_2 = self.radial.a0(self.freq_2) * np.sqrt(self.radial.vis2)
        ell = int(2)
        eea = self.sphr_lm(2, self.radial.inc)
        try:
            print(self.l2_split)
        except:
            self.l2_split = np.ones(len(self.freq_2)) * 0.15
        # Need to check way that visibility added in is corect!
        for i in range(len(self.freq_2)):
            mod += self.lor(self.freq_2[i],
                                   self.radial.width_0[i],
                                   self.amp_2[i], eea[0])
            mod += self.lor(self.freq_2[i] - self.l2_split[i],
                                  self.radial.width_0[i],
                                  self.amp_2[i], eea[1])
            mod += self.lor(self.freq_2[i] + self.l2_split[i],
                                  self.radial.width_0[i],
                                  self.amp_2[i], eea[1])
            mod += self.lor(self.freq_2[i] - 2.0 * self.l2_split[i],
                                  self.radial.width_0[i],
                                  self.amp_2[i], eea[2])
            mod += self.lor(self.freq_2[i] + 2.0 * self.l2_split[i],
                                  self.radial.width_0[i],
                                  self.amp_2[i], eea[2])

        self.model += (mod * self.eta_sq())

    def sphr_lm(self, l, theta):
        ell = int(l)
        amp = np.zeros(ell + 1)
        for mdx, m in enumerate(range(0, ell+1)):
            H = (factorial(ell - abs(m))/factorial(ell + abs(m))) \
                * legendre(m, ell, np.cos(theta*np.pi/180))**2
            amp[mdx] = H
        return amp

    def lor(self, frequency, width, amp, vis):
        height = 2.0 * amp ** 2.0 / (np.pi * width)
        x = 2.0 * (self.radial.f - frequency) / width
        return height * vis / (1.0 + x**2)

    def eta_sq(self):
        """
        Compute sinc^2 modulation from sampling
        """
        return np.sinc(self.radial.f/(2.0*self.radial.f.max()))**2.0

if __name__=="__main__":

    # Let's check it works for some example values!
    mission = 'Kepler'
    dnu = 17.4
    numax = 243.0
    dpi = 80.0
    epsg = 0.0
    gsplit = 0.5
    R = 0.0
    inc = 45.0
    q = 0.2

    Teff = 4840
    kmag = 12.44

    global_params_header = ['mission: Kepler=0/TESS=1', 'dnu', 'numax', 'dpi', 'epsg', 'gsplit',
                            'R', 'inc', 'q', 'Teff', 'kmag']

    if mission == 'Kepler':
        global_params = [0, dnu, numax, dpi, epsg, gsplit, R, inc, q, Teff,
                     kmag]
    elif mission == 'TESS':
        global_params = [1, dnu, numax, dpi, epsg, gsplit, R, inc, q, Teff,
                     kmag]

    if mission == 'Kepler':
        dt = 29.4*60.0
        T = 4.0 * 365.25 * 86400.0
    elif mission == 'TESS':
        dt = 30.0*60.0
        obs_time = raw_input("How long is observing time (long/short)?\n")
        if obs_time == 'long':
            T = 356.2 * 86400.0
        elif obs_time == 'short':
            T = 164.4 * 86400.0

    nyq = 1.0 / (2.0 * dt) * 1e6
    freq = np.arange(0, nyq, (1.0 / T)*1e6)


    print("REMEMBER TO INCLUDE 0.85 FACTOR IN ALL AMPLITUDES AND CHANGE IN MODE VISIBILITIES FOR TESS MISSION!!!!")
    print("PROBLEM WITH CALCULATING l=1 HEIGHTS, TALK TO BILL!!!")
    # Set up class for scaling relations
    backg = Background(freq, nyq, dnu, numax, dt, kmag, N=1, \
                       mission=mission)
    model = backg()
    try:
        freq_0 = np.loadtxt('test_radial.txt')
    except:
        freq_0 = []
    modes = RadialModes(freq, model, Teff, kmag, dnu, numax, dpi, epsg, q, gsplit, R, \
                  inc, dt, T, N=1, mission='Kepler')
    modes.create_radial_modes(freq_0)
    l2 = QuadrupoleModes(modes,[])
    l2.create_quadrupole_modes()

    # Create power spectrum -> multiply by chi2 2dof noise
    power = model * -1.0 * np.log(np.random.uniform(0,1,len(freq)))

    pl.plot(freq, power, 'k')
    pl.plot(freq, model, 'b')
    pl.xlim([1, freq.max()])
    pl.xlabel(r'Frequency ($\mu$Hz)')
    pl.ylabel(r'PSD (ppm$^{2}\mu$Hz$^{-1}$)')
    pl.show()

    np.savetxt('test_radial.txt', modes.freq_0)
