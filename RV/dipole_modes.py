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

class DipoleModes(object):
    """
    Class to compute radial modes from inputs
    """
    def __init__(self, radial, _model, _freq_1 = []):

        # calc nominal p-mode frequency
        self.radial = radial
        self.model = _model
        self.freq = self.radial.freq
        self.d01 = self.radial.d01_scaling()
        ###
        self.width_1 = self.radial.width_0
        # Make sure mixed modes aren't generated beyong the nyquist frequency
        self.d01 = np.array([self.d01] * len(self.radial.freq_0)) #self.d01[:len(self.radial.freq_0)]
        self.nom_p_freq = self.radial.freq_0 + self.d01
        sel = np.where(self.nom_p_freq >= (self.radial.nyq*1e6 - 1.0))
        self.nom_p_freq = np.delete(self.nom_p_freq, sel)
        # Ensure the same the number of orders of l=1 corresponds to l=0,2
        if len(self.radial.freq_0) != len(self.nom_p_freq):
            self.radial.freq_0 = np.delete(self.radial.freq_0, -1)
            self.radial.amp_0 = np.delete(self.radial.amp_0, -1)
        self.p_tot = 0
        # Accept frequencies as input
        if len(_freq_1) > 0:
            self.mixed_full = _freq_1
        else:
            self.mixed_full = None

    def calc_Q(self):
        """
        Calculate the factor Q -> ratio of l=1 intertia to l=0 inertia.
        Also given in terms of the observable xi, Q = 1 / (1 - xi).
        Where xi is given by Deheuvels et al. 2015
        """
        return 1.0 / (1.0 - self.xi_full)

    def calc_xi(self, mixed, k):
        """
        Calculate xi from Deheuvels et al. 2015 - ratio of core to envelope
        contributions to total inertia.
        """
        numerator = np.cos(np.pi * ((1.0 / (mixed*1e-6 * self.radial.dpi)) - self.radial.epsg))**2.0
        denominator = np.cos(np.pi * (mixed - self.nom_p_freq[k])/self.radial.dnu)**2.0
        xi = (1.0 + (1.0 / self.radial.q) * numerator/denominator * \
             (mixed*1e-6)**2.0*self.radial.dpi / (self.radial.dnu*1e-6))**-1.0
        # Try using Goupil et al. 2013 to check consistency
        #alpha0 = self.radial.dnu * 1e-6 * self.radial.dpi
        #chi = 2.0 * mixed / self.radial.dnu * np.cos(np.pi / (mixed*1e-6*self.radial.dpi))
        #xi = 1.0 / (1.0 + alpha0 * chi**2.0)

        

        #pl.plot(xi1, 'k')
        #pl.plot(xi, 'r')
        #pl.show()

        return xi


    def calc_heights(self, mixed, xi):
        """
        Calculate the mode heights from Bill's book: (per order)

        """
        self.heights = []
        # Again, work out for each radial order
        for i in range(len(self.radial.freq_0)):
            Q = 1.0 / (1.0 - xi[i])
            numerator = 2.0 * self.radial.vis1 * self.radial.T * (self.radial.a0(self.nom_p_freq[i]))**2.0
            # STILL NEED TO CALCULATE Q IN TERMS OF
            denominator = np.pi * self.radial.T * self.width_1[i]*1e-6 + 2.0 * Q 
            self.heights.append((numerator / denominator)*1e-6)
            #tmp_0 = numerator/denominator * 1e-6
            #tmp_1 = 2.0 * (self.radial.a0(self.nom_p_freq[i]))**2.0 / (np.pi * self.width_1[i] * Q ** -1.0)

            

            #pl.plot(tmp_1/tmp_0, 'rD')
            #pl.yscale('log')
            #pl.plot(xi[i])
            #pl.plot(1-xi[i])
            #pl.plot(xi[i])
            #pl.plot(np.pi * self.radial.T * self.width_1[i]*1e-6 * np.ones(len(Q)), 'r--')
            #pl.plot(2.0*Q, 'k--')
            #pl.ylabel(r'$\pi \mathrm{T} \Gamma_{0}(\nu_{1})$')
            #pl.xlabel(r'Mixed mode number')
            #pl.show()
        #pl.title(r"$\xi$")
        #pl.show()
        print("Heights: ", self.heights)

    def calc_splittings(self, mixed, k):
        """
        Calculate mode splittings using Deheuvels et al. 2015
        """
        xi = self.calc_xi(mixed, k)
        #pl.title(r'$\xi$')
        return xi, (xi * (1.0 - 2.0*self.radial.R) + 2.0 * self.radial.R) * self.radial.gsplit
        #alpha_0 = (self.dnu * 1e-6) * self.dpi
        #chi = 2.0 * mixed *1e6 / self.dnu * np.cos(np.pi / (self.dpi * mixed))
        #xi = 1.0 / (1.0 + alpha_0 * chi**2)
        #pl.plot(mixed, xi)
        #pl.show()
        #return xi, self.gsplit * (xi + (1 - 2.0*self.R) + 2.0*self.R)


    def calc_dipole_mixed_freqs(self):
        """
        Calculate the frequencies of mixed l=1 modes
        """

        # Need to calculate frequencies, inertia and splitting separately!

        # Calculate for each radial order and append to list, store inertia as well
        self.mixed_full = []
        self.xi_full = []
        self.split_full = []
        self.dnu = self.radial.dnu
        bw = self.freq[1] - self.freq[0]
        for i in range(len(self.radial.freq_0)):
            zero = self.radial.freq_0[i]
            pone = self.nom_p_freq[i]

            #Calculate the mixed mode frequencies ...
            nu = np.arange(zero - 0.5 * self.dnu, zero + 1.5 * self.dnu, bw) * 1e-6
            lhs = np.pi * (nu - (pone * 1e-6)) / (self.dnu * 1e-6)
            rhs = np.arctan(self.radial.q * np.tan(np.pi / (self.radial.dpi * nu) - self.radial.epsg))
            mixed1 = np.zeros(1000)
            counter = 0
            for j in np.arange(0, nu.size-1):
                if (lhs[j] - rhs[j] < 0) and (lhs[j+1] - rhs[j+1] > 0):
                    mixed1[counter] = nu[j]
                    counter += 1
            mixed1 = mixed1[:counter]
            idx = np.where(mixed1*1e6 > (self.radial.nyq*1e6 - 1.0))
            mixed1 = np.delete(mixed1, idx)
            #xi, split = self.calc_splittings(mixed1, i)
            # Append values to lists
            #self.split_full.append(split)
            self.mixed_full.append(mixed1*1e6)
            #self.xi_full.append(xi)

    def calc_mixed_splittings(self):
        """
        Calculate mixed mode splittings
        """
        self.split_full = []
        self.xi_full = []
        for i in range(len(self.radial.freq_0)):
            xi, split = self.calc_splittings(self.mixed_full[i], i)
            self.split_full.append(split)
            self.xi_full.append(xi)

    def create_dipole_mixed_modes(self):
        """
        Create dipole mixed modes
        """
        # Creates frequencies, splittings and inertia
        if self.mixed_full is None:
            self.calc_dipole_mixed_freqs()
        mod = np.zeros(len(self.radial.f))
        self.calc_mixed_splittings()
        # Calculate Q from xi
        self.xi_full = np.array(self.xi_full)
        self.split_full = np.array(self.split_full)
        self.mixed_full = np.array(self.mixed_full)

        # Calculate l=2 splitting
        self.l2_split = []
        for i in range(len(self.split_full)):
            self.l2_split.append(np.min(self.split_full[i])/2.0)
        self.l2_split = np.array(self.l2_split)

        self.Q = 1.0 / (1.0 - self.xi_full)
        #pl.title('$Q^{-1}$')
        #for i in range(len(self.xi_full)):
        #    #pl.plot(self.mixed_full[i], self.xi_full[i]**-1.0)
        #    pl.plot(self.mixed_full[i], self.Q[i]**-1.0)
        #pl.show()

        #pl.title('$\ell=1$ widths')
        #for i in range(len(self.Q)):
        #    pl.plot(self.mixed_full[i], self.Q[i]**-1.0 * self.width_1[i])
        #pl.axhline(2.0 * (self.f[1]-self.f[0]), color='r', linestyle='--')
        #pl.show()

        self.calc_heights(self.mixed_full, self.xi_full)
        #pl.title('$\ell=1$ heights')
        #for i in range(len(self.Q)):
        #    pl.plot(self.f, self.a0(self.f)**2.0 * 2.0 / np.pi / self.width_1[i], 'r')
        #    pl.plot(self.mixed_full[i], self.heights[i])
        #pl.show()
        ell = int(1)
        eea = self.sphr_lm(1, self.radial.inc)
        print("INCLINATION ANGLE: ", self.radial.inc)
        self.dipole_width = []
        for i in range(len(self.radial.freq_0)):
            for j in range(len(self.mixed_full[i])):
                mod += self.lor(self.mixed_full[i][j],
                                self.width_1[i] * (self.Q[i][j] ** -1.0),
                                self.heights[i][j], eea[0])
                mod += self.lor(self.mixed_full[i][j] - self.split_full[i][j],
                                self.width_1[i] * (self.Q[i][j] ** -1.0),
                                self.heights[i][j], eea[1])
                mod += self.lor(self.mixed_full[i][j] + self.split_full[i][j],
                                self.width_1[i] * (self.Q[i][j] ** -1.0),
                                self.heights[i][j], eea[1])
                self.dipole_width = np.append(self.dipole_width, self.width_1[i] * (self.Q[i][j] ** -1.0))
        self.model += (mod * self.eta_sq())
        self.p_tot = np.sum(mod * self.eta_sq())
        print("TOTAL POWER IN l=1: ", self.p_tot)

    def sphr_lm(self, l, theta):
        ell = int(l)
        amp = np.zeros(ell + 1)
        for mdx, m in enumerate(range(0, ell+1)):
            H = (factorial(ell - abs(m))/factorial(ell + abs(m))) \
                * legendre(m, ell, np.cos(theta*np.pi/180))**2
            amp[mdx] = H
        return amp

    def lor(self, frequency, width, height, vis):
        x = 2.0 * (self.radial.f - frequency) / width
        return height * vis / (1.0 + x**2)

    def model(self, params):
        freqs, amp, width, split, inc = params
        eea = self.sphr_lm(1, inc)
        mode = self.lorentzian(freqs, width, amp, eea[0])
        mode += self.lorentzian(freqs-split, width, amp, eea[1])
        mode += self.lorentzian(freqs+split, width, amp, eea[1])
        return mode

    def __call__(self, params):
        freqs, amp, width, split, inc = params
        mod = np.zeros(len(self.f))
        for i in range(len(freqs)):
            mod += self.model([freqs[i], amp[i], width[i], \
                               split[i], inc[i]])
        return mod

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


    #print("REMEMBER TO INCLUDE 0.85 FACTOR IN ALL AMPLITUDES AND CHANGE IN MODE VISIBILITIES FOR TESS MISSION!!!!")
    #print("PROBLEM WITH CALCULATING l=1 HEIGHTS, TALK TO BILL!!!")
    # Set up class for scaling relations
    backg = Background(freq, nyq, dnu, numax, dt, kmag, N=1, \
                       mission=mission)
    model = backg()
    modes = RadialModes(freq, model, Teff, kmag, dnu, numax, dpi, epsg, q, gsplit, R, \
                  inc, dt, T, N=1, mission='Kepler')
    modes.create_radial_modes()
    l_1 = DipoleModes(modes, modes.model, [])
    l_1.create_dipole_mixed_modes()


    # Create power spectrum -> multiply by chi2 2dof noise
    power = model * -1.0 * np.log(np.random.uniform(0,1,len(freq)))

    pl.plot(freq, power, 'k')
    pl.plot(freq, model, 'b')
    pl.xlim([1, freq.max()])
    pl.xlabel(r'Frequency ($\mu$Hz)')
    pl.ylabel(r'PSD (ppm$^{2}\mu$Hz$^{-1}$)')
    pl.show()
