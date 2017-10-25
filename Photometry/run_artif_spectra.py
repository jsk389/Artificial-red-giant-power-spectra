# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from background import Background
import numpy as np
import matplotlib.pyplot as pl
from scalings import Scalings
from radial_modes import RadialModes
from dipole_modes import DipoleModes
from l_2_modes import QuadrupoleModes
import os
from scipy import integrate


if __name__=="__main__":

    # Let's take the approximate values for KOI-3890
    mission = 'Kepler'
    dnu = 9.57
    numax = 103.4
    dpi = 74.9
    epsg = 0.0
    gsplit = 0.4
    R = 0.0
    inc = 90.0
    q = 0.7

    Teff = 4840
    kmag = 12.44
    # If Henv is not given then it will be calculated automatically
#    Henv = 135.0

    global_params_header = ['mission: Kepler=0/TESS=1', 'dnu', 
                            'numax', 'dpi', 'epsg', 'gsplit',
                            'R', 'inc', 'q', 'Teff', 'kmag']

    if mission == 'Kepler':
        global_params = [0, dnu, numax, dpi, epsg, gsplit, 
                         R, inc, q, Teff, kmag]
    elif mission == 'TESS':
        global_params = [1, dnu, numax, dpi, epsg, gsplit, 
                         R, inc, q, Teff, kmag]
    # Cadence
    dt = 29.4*60.0
    # Length of timeseries
    T = 4.0 * 365.25 * 86400.0
    # Nyquist
    nyq = 1.0 / (2.0 * dt) * 1e6
    # Frequency array
    freq = np.arange(0, nyq, (1.0 / T)*1e6)
    # Bin width
    bw = (1.0/T)*1e6

    # Set up class for scaling relations
    backg = Background(freq, nyq, dnu, numax, dt, kmag, mission)
    model = backg()
    backg_model = model.copy()
    # Appends frequencies to a list - terrible idea but this was written a long time ago!
    freq_0 = []
    l_0 = RadialModes(freq, model, Teff, kmag, dnu, numax, dpi, epsg, q, gsplit, R, \
                  inc, dt, T, mission, None)#, Henv)
    l_0.create_radial_modes(freq_0)
    # Don't need to read in l=1 freqs as should be the same provided parameters
    # such as dpi, q etc. are saved
    l_1 = DipoleModes(l_0, l_0.model, [])
    l_1.create_dipole_mixed_modes()
    # Compute l=2 frequencies etc.
    freq_2 = []
    l_2 = QuadrupoleModes(l_0, l_1, l_1.model, freq_2)
    l_2.create_quadrupole_modes()


    # Create power spectrum -> multiply by chi2 2dof noise
    power = model * -1.0 * np.log(np.random.uniform(0,1,len(freq)))

    # Save l=0,1,2 parameters
    l0_freqs = np.array(l_0.freq_0).flatten()
    l0_amp = np.array(l_0.amp_0).flatten()
    # l=1
    l1_freqs = [item for sublist in l_1.mixed_full for item in sublist]
    l1_width = l_1.dipole_width
    l1_split = [item for sublist in l_1.split_full for item in sublist]
    l1_angle = np.ones(len(l1_freqs)) * inc
    l1_heights = [item for sublist in l_1.heights for item in sublist]
    idxs = np.array(l1_freqs)/bw
    l1_backg = backg_model[idxs.astype(int)]
    # l=2
    l2_freqs = np.array(l_2.freq_2).flatten()
    l2_amp = np.array(l_2.amp_2).flatten()

    np.savetxt('KOI-3890_spec.power', np.c_[freq, power])
    pl.plot(f, p, 'k')
    pl.plot(f, model, 'r')
    #pl.xscale('log')
    #pl.yscale('log')
    pl.show()

  
    directory = 'Red_split_spectra/'
    # Save data
    np.savetxt(directory+str(i)+'_global.txt', global_params, header=str(global_params_header))
    ## Save radial mode parameters
    np.savetxt(directory+str(i)+'_radial.txt', np.c_[l0_freqs, l0_amp])
    ## Save l=1 mode parameters
    np.savetxt(directory+str(i)+'_l1.txt', np.c_[l1_freqs, l1_heights, l1_width, l1_split, l1_angle, l1_backg])
    ## Save l=2 mode parameters
    np.savetxt(directory+str(i)+'_l2.txt', np.c_[l2_freqs, l2_amp])
    # Save out power spectrum    
    np.savetxt(directory+str(i)+'.txt', np.c_[freq, power])
    # Save out model
    np.savetxt(directory+str(i)+'_model.txt', np.c_[freq, model])



