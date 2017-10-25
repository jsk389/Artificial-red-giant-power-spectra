# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from background import Background
import numpy as np
import matplotlib.pyplot as pl
from scalings import Scalings

class RadialModes(Scalings):
    """
    Class to compute radial modes from inputs
    """
    def __init__(self, _f, _model, *args, **kwargs):

        # Use multiple inheritance so that RadialMode class inherits all parameters
        # from Scalings class
        super(RadialModes, self).__init__(*args, **kwargs)
        self.model = _model
        self.N = self.n_env
        # Initialise model with frequency array
        self.f = _f
        self.widths_0 = self.gamma_0
        self.d02 = self.d02_scaling()
        #self.n_env = self.n_env()
        self.epsilon = self.epsilon()
        self.amax = self.amax_scaling()
        self.gamma_0 = self.gamma_0
        self.a0 = self.a0()

    def radial_mode_freqs(self):
        """
        Create n_env radial modes at given frequency
        """
        n_max = np.floor((self.numax / self.dnu) - self.epsilon)
        print("NUMBER OF RADIAL ORDERS: ", self.n_env)
        #self.n_env += 7
        self.freq_0 = np.zeros(self.n_env)
        # Go from nmax - nenv/2 to nmax + nev/2
        # Work out minimum and maximum radial order
        self.nmin = n_max - len(self.freq_0)//2
        self.nmax = n_max + len(self.freq_0)//2
        # Create array of n to work over
        self.n = np.arange(self.nmin, self.nmax, 1)
        # Set curvature term for l=0
        self.alpha_0 = 0.008 #np.random.normal(0.008, 0.001)

        tmp_0 = self.n + (0.0/2.0) + self.epsilon + \
               (self.alpha_0 / 2.0 * (self.n - (self.numax / self.dnu))**2.0)

        self.freq_0 = tmp_0 * self.dnu

        #for i in range(-len(self.freq_0)//2, len(self.freq_0)//2):
        #    self.freq_0[i] = ((n_max - i) + \
        #                      self.epsilon) * self.dnu
        #    self.freq_0[i]
        #    # NO CURVATURE INCLUDED!
        return np.sort(self.freq_0)

    def radial_mode_amplitudes(self):
        """
        Calculate radial mode amplitudes
        """
        return self.a0(self.freq_0)

    def width_model(self, x, theta):
        alpha, gamma, gamma_dip, w_dip = theta
        return np.exp((alpha * np.log(x) + np.log(gamma)) - (np.log(gamma_dip) / (1. + ((2.0*np.log(x))/(np.log(w_dip)))**2.0)))

    def radial_mode_widths(self):
        """
        Calculate radial mode widths using fitted expression to low luminosity red giants in angle of inclination work
        """
        params = [2.96, 0.52, 7.5, 0.78]
        return self.width_model(self.freq_0/self.numax, params)
        #factor = (self.freq_0.max()/(self.nyq*1e6))**0.2
        #return (self.gamma_0()*factor) * (self.freq_0 / (self.nyq*1e6))**0.2

    def create_radial_modes(self, fre = []):
        """
        Create radial modes
        """
        if len(fre) == 0 :
            self.freq_0 = self.radial_mode_freqs()
        else:
            self.freq_0 = fre
        # Remove any super-nyquist frequencies
        sel = np.where(self.freq_0 >= (self.nyq*1e6 - 1.0))
        self.freq_0 = np.delete(self.freq_0, sel)
        # Reset number of radial orders in envelop to account for any deleted due to
        # being above nyquist
        self.n_env = len(self.freq_0)
        print("NUMBER OF RADIAL MODE FREQS: ", len(self.freq_0))

        # Calculate radial mode amplitudes
        self.amp_0 = self.radial_mode_amplitudes()

        # Calculate radial mode widths and parameterise as fn of freq.
        self.width_0 = self.radial_mode_widths()
        #print("RADIAL MODE WIDTH (1): ", self.width_0)
        # Generate model
        mod = np.zeros(len(self.f))
        print(len(mod), len(self.freq), len(self.f))
        for i in range(len(self.freq_0)):
            print(i)
            mod += self.lorentzian(self.freq_0[i],
                                          self.width_0[i],
                                          self.amp_0[i], 1.0)
            print("done")
        print(len(mod), len(self.eta_sq()))
        print("TOTAL POWER IN l=0: ", np.sum(mod * self.eta_sq()))
        self.model += (mod * self.eta_sq())

    def lorentzian(self, frequency, width, amplitude, vis):
        x = 2.0 * (self.f - frequency) / width
        height = (2.0 * amplitude**2.0 * self.T) / (np.pi * width * 1e-6 * self.T + 2)
        height *= 1e-6
        #height = (2.0 * amplitude**2.0) / (np.pi * width)
        return height * vis / (1.0 + x**2)

    def eta_sq(self):
        """
        Compute sinc^2 modulation from sampling
        """
        return np.sinc(self.f/(2.0*self.f.max()))**2.0

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


    # Create power spectrum -> multiply by chi2 2dof noise
    power = model * -1.0 * np.log(np.random.uniform(0,1,len(freq)))

    pl.plot(freq, power, 'k')
    pl.plot(freq, model, 'b')
    pl.xlim([1, freq.max()])
    pl.xlabel(r'Frequency ($\mu$Hz)')
    pl.ylabel(r'PSD (ppm$^{2}\mu$Hz$^{-1}$)')
    pl.show()

    np.savetxt('test_radial.txt', modes.freq_0)
