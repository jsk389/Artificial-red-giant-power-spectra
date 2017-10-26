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
    loc='./Artificial_Spectra/'
    if not os.path.isdir(loc):
        os.makedirs(loc)

    folder=loc+'Subgiant_numax1000/'
    if not os.path.isdir(folder):
        os.makedirs(folder)
    star='Subgiant'
    
    # Access to kic number, kepler magnitude, teff and dnu
    # Cadence
    dt = 200.0
    # Length of observation
    T = 100 * 86400.0
    # Nyquist
    nyq = 1.0 / (2.0 * dt) * 1e6
    # Frequency array
    freq = np.arange(0, nyq, (1.0 / T)*1e6)
    # Bin width
    bw = (1.0/T)*1e6
    # Kepler mag (This functions as kepmag for Kepler and I-band mag for TESS)
    kmag = 5.0
    # Effective temperature
    Teff = 5000
    # numax
    numax = 500
    # dpi1
    dpi = 100
    # dnu
    dnu = 0.276 * numax ** 0.754
    # Henv - not calculated in code - easier to give it here using scaling relation - only needed for oscillations, not background
    Henv = 2.03e7 * numax ** -2.38
    # Extra mixed mode parameters
    epsg = 0.0
    q = 0.17
    gsplit = 0.3
    R = 0.0

    numax=1150
    dnu=0.251*numax**0.751
    Teff=5702
    dpi=500


    # Angle of inclination
    inc = 90.0
    # This is for Kepler red giants therefore set mission keyword to Kepler
    mission = 'SONG'
    n = []

    global_params = np.array(['', dnu, numax, dpi, epsg, gsplit, 
                              R, inc, q, Teff, Henv, kmag]).astype(str)
    # Set up class for scaling relations
    print('Creating background')
    backg = Background(freq, nyq, dnu, numax, dt, kmag, Teff, mission)
    model = backg()
    backg_model = model.copy()
    freq_0 = []
    print('l=0')
    l_0 = RadialModes(freq, model, Teff, kmag, dnu, numax, dpi, epsg, q, gsplit, R, \
                  inc, dt, T, mission, Henv)
    l_0.create_radial_modes(freq_0)
    # Don't need to read in l=1 freqs as should be the same provided parameters
    # such as dpi, q etc. are saved
    print('l=1')
    l_1 = DipoleModes(l_0, l_0.model, [])
    l_1.create_dipole_mixed_modes()
    #try:
    #    freq_2 = np.loadtxt('Kep56-reference_l2.txt', usecols=(0,))
    #except:
    freq_2 = []
    print('l=2')
    l_2 = QuadrupoleModes(l_0, l_1, l_1.model, freq_2)
    l_2.create_quadrupole_modes()


    # Create power spectrum -> multiply by chi2 2dof noise
    power = model * -1.0 * np.log(np.random.uniform(0,1,len(freq)))

    # Save l=0,1,2 parameters
    l0_freqs = np.array(l_0.freq_0).flatten()
    l0_amp = np.array(l_0.amp_0).flatten()
    l0_width = np.array(l_0.width_0).flatten()
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
	
    pl.plot(freq,power)
    #pl.plot(freq,model)
    pl.show()


    np.savetxt(folder+str(star)+'_ps.pow', np.c_[freq, power])
    np.savetxt(folder+str(star)+'_model.pow', np.c_[freq, model])

    np.savetxt(folder+str(star)+'_global.txt', global_params, fmt="%s")#, header=str(global_params_header))
    # Save radial mode parameters
    np.savetxt(folder+str(star)+'_radial.txt', np.c_[l0_freqs, l0_amp, l0_width])
    # Save l=1 mode parameters
    np.savetxt(folder+str(star)+'_l1.txt', np.c_[l1_freqs, l1_heights, l1_width, l1_split, l1_angle, l1_backg])
    # Save l=2 mode parameters
    np.savetxt(folder+str(star)+'_l2.txt', np.c_[l2_freqs, l2_amp])

    
