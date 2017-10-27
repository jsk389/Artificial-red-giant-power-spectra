#!/usr/bin/env python3

import gatspy.periodic as gp
import matplotlib.pyplot as plt
import numpy as np

"""
This module contains the class to generate and (optionally) corrupt a timeseries
from a power spectrum.
"""

class MyException(Exception):
    pass

class CorruptSpectra(object):
    """
    A class designed to generate and corrupt power spectra/timeseries
    """

    def __init__(self, _freq, _power, _obs_length, observing_window=None, corrupt=True):
        """
        Initialise class

        Creates a new :class:`CorruptSpectra` instance.

        :param _freq: Frequency array of star whose data is to be corrupted.
        :type _freq: array.

        :param _power: Power array of star whose data is to be corrupted.
        :type _power: array.

        :param _obs_length: Length of the observations in days - this is converted
                            into seconds later.
        :type _obs_length: float.

        :params observing_window: Observing window (per day), to be used in
                                   conjuction with corrupt argument. Defaults
                                   to None.
        :type observing_window: float

        :params corrupt: (optional) Whether or not to corrupt the timeseries
                          with a window function. Defaults to True.
        :type corrupt: bool

        """
        self.freq = _freq
        self.power = _power
        self.obs_length = _obs_length * 86400.0
        self.observing_window = observing_window
        self.corrupt = corrupt

        # Compute cadence and total observing length
        self.dt = 1e6/(2*self.freq[-1])
        # Define bin width
        self.bw = self.freq[1]-self.freq[0]
        self.total_length = 1e6/self.bw# in s

        if (self.corrupt==True) and (self.observing_window == None):
            raise MyException("Corrupt keyword is True but no observing window is given!")

    def rebin(self, x, r):
        """
        Rebin an array x using a window of length r

        :param x: Array to be rebinned
        :type x: array

        :param r: Number of bins to rebin over
        :type r: int
        """
        m = len(x) // int(r)
        return x[:m*r].reshape((m,r)).mean(axis=1)

    def create_window(self, time, observing_window):
        """
        Create a window function that mimicks an observing window every 24 hours

        :param time: Time array to create window from
        :type time: array

        :param observing_window: Length of observing window (in hours)
        :type observing_window: float
        """
        tmp = np.ones_like(time)
        tmp[(time/3600)%24<(24-observing_window)] = 0
        self.fill = len(tmp[tmp == 0]) / len(tmp)
        return tmp

    def compute_power_spectrum(self, time, flux):
        """
        Compute the power spectrum of a given timeseries

        :param time: Array containing the time array
        :type time: array

        :param flux: Array containing flux values
        :type flux: array
        """
        time=time-time[0]
        if time[1] < 1:
            time *= 86400.0
        nyq=1./(2*np.median(np.diff(time)))
        df=1./time[-1]

        f,p=gp.lomb_scargle_fast.lomb_scargle_fast(time,flux,f0=0,df=df,Nf=1*(nyq/df))

        lhs=(1./float(len(time)))*np.sum(flux**2)
        rhs= np.sum(p)
        ratio=lhs/rhs
        p*=ratio/(df*1e6)#ppm^2/uHz
        fill = float(len(np.where(flux != 0.0)[0])) / float(len(flux))
        p /= fill
        f*=1e6
        return f, p

    def normalise_timeseries(self, time, flux):
        """
        Normalise the timeseries according to Parseval's theorem

        :param time: Array containing the time array
        :type time: array

        :param flux: Array containing flux values to normalise
        :type flux: array
        """

        lhs = (1 / len(time)) * np.sum(flux**2)
        # Power in ppm^2uHz^-1 -> needs to be in ppm^2bin^-1 for parseval!
        rhs = np.sum(self.power * self.bw)
        ratio = lhs/rhs

        # Divide by square root to put ratio into amplitude!!!!!
        flux /= np.sqrt(ratio)
        return time, flux

    def generate_timeseries(self):
        """
        Generate a timeseries from the power spectrum following the method
        given in ... (NEED REFERENCE!)
        """
        # Convert power into per bin from per uHz
        power_per_bin = self.power * self.bw
        # Create a set of random numbers for generation of timeseries
        real_comp = np.random.normal(0, 1, len(power_per_bin)) * \
                                     np.sqrt(power_per_bin/2.0)
        imag_comp = np.random.normal(0, 1, len(power_per_bin)) * \
                                     np.sqrt(power_per_bin/2.0)
        # Define array for inverse fft
        inv_fft = np.zeros(len(power_per_bin), dtype=complex)
        for i in range(len(inv_fft)):
            inv_fft[i] = real_comp[i] + 1j*imag_comp[i]

        x = np.fft.irfft(inv_fft) * len(power_per_bin)
        # Create time array
        time = np.arange(0, self.total_length, self.dt)#in seconds
        self.time_full, self.flux_full = self.normalise_timeseries(time, x)

        # Corrupt timeseries
        if self.corrupt == True:
            self.window = self.create_window(self.time_full, self.observing_window)
            self.flux_full_uncorrupted = self.flux_full.copy()
            self.flux_full *= self.window

        # Cut down timeseries to desired length
        flux_uncorrupted = self.flux_full_uncorrupted[self.time_full < self.obs_length]
        flux = self.flux_full[self.time_full < self.obs_length]
        time = self.time_full[self.time_full < self.obs_length]

        return time, flux, flux_uncorrupted



if __name__ == "__main__":

    star='../RV/Artificial_Spectra/Subgiant_numax1000/Subgiant_ps.pow'
    fin, pin = np.loadtxt(star, unpack=True)
    obs_length = 200
    obs_window = 8
    # _freq, _power, _obs_length, observing_window=None, corrupt=True
    corrupt = CorruptSpectra(fin, pin, obs_length,
                             observing_window=obs_window,
                             corrupt=True)

    time, x, x_uncorrupted = corrupt.generate_timeseries()

    # Generate power spectrum on truncated timeseries
    freq, power = corrupt.compute_power_spectrum(time, x_uncorrupted)

    #np.savetxt(mypath+'/gapless_ps.pow',np.c_[freq,power])#,fmt='%10.4f')
    # Compare power spectra
    plt.plot(fin, pin, 'k',alpha=0.6,label='Original')
    plt.plot(freq, power, 'r', alpha=0.5,label='PS-$>$TS-$>$PS')
    plt.legend(loc='best')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    # Generate power spectrum with window applied
    freq, power = corrupt.compute_power_spectrum(time, x)

    # Plot to compare input spectra to chopped down and gapped data
    plt.plot(fin, pin, 'k', label='Original')
    plt.plot(freq, power, 'r',label='butchered')
    plt.legend(loc='best')
    plt.xlabel('Frequency ($\mu$Hz)', fontsize=18)
    plt.ylabel(r'PSD (m$^{2}$s$^{-2}\mu$Hz$^{-1}$)', fontsize=18)
    plt.show()

    #Fill on data
    print('Fill (%):', 100*corrupt.fill)
