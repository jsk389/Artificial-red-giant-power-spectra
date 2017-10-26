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

    # Let's take the approximate values for KOI-3890
    mission = 'Kepler'
    dnu = 9.57
    numax = 103.4
    dpi = 74.9
    epsg = 0.0
    gsplit = 0.4
    R = 0.0
    inc = 90.0
    q = 0.01#coupling
    Teff = 4840
    kmag = 12.44

    numax=1150
    dnu=0.251*numax**0.751
    Teff=5702
    dpi=500

    # If Henv is not given then it will be calculated automatically
#    Henv = 135.0

    global_params_header = ['mission: Kepler=0/TESS=1', 'dnu', 
                            'numax', 'dpi', 'epsg', 'gsplit',
                            'R', 'inc', 'q', 'Teff', 'kmag']

    global_params = [str(mission), dnu, numax, dpi, epsg, gsplit, 
                         R, inc, q, Teff, kmag]
    # Cadence
    dt = 29.4*60.0
    dt= 58.85
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
    ###############################################################################
    import pyfits
    from os.path import expanduser
    home=expanduser('~')
    i=home+'/Dropbox/PhD/Python_Codes/kplr006442183_kasoc-psd_slc_v1.fits'

    data=pyfits.open(i)
    kic=data[0].header['KEPLERID']
    obs=data[0].header['OBSMODE']
    f,p=data[1].data['FREQUENCY'],data[1].data['PSD']

    pl.figure()
    pl.plot(freq, power, 'k')
    pl.vlines(l2_freqs,0,500,linestyle='--',color='b',alpha=0.5)
    pl.vlines(l1_freqs,0,500,linestyle=':',color='r',alpha=0.5)
    pl.vlines(l0_freqs,0,500,linestyle='--',color='g',alpha=0.5)
    
    pl.figure()
    pl.plot(f,p, 'r',alpha=0.8)
    #pl.xscale('log')
    #pl.yscale('log')
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


