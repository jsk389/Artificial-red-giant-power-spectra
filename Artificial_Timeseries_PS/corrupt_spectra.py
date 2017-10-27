#!/usr/bin/env python
from __future__ import print_function,division
import gatspy.periodic as gp
import matplotlib.pyplot as plt
import numpy as np
from os.path import expanduser
import sys,os
from scipy.optimize import curve_fit

def med_filt(x, y, dt=4.):
    """
    De-trend a light curve using a windowed median.
    """
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    assert len(x) == len(y)
    r = np.empty(len(y))
    for i, t in enumerate(x):
        inds = (x >= t - 0.5 * dt) * (x <= t + 0.5 * dt)
        r[i] = np.nanmedian(y[inds])
    return r

def rebin(x, r):
    m = len(x) // r
    return x[:m*r].reshape((m,r)).mean(axis=1),r

def ps(time,flux):
    time=time-time[0]

    nyq=1./(2*(time[1]-time[0]))
    df=1./time[-1]
    #print('Nyq:',1e6*nyq, 'Resolution:',1e6*df)
    f,p=gp.lomb_scargle_fast.lomb_scargle_fast(time,flux,f0=0,df=df,Nf=1*(nyq/df))

    lhs=(1./float(len(time)))*np.sum(flux**2) 
    rhs= np.sum(p)
    ratio=lhs/rhs
    p*=ratio/(df*1e6)#ppm^2/uHz
    fill = float(len(np.where(flux != 0.0)[0])) / float(len(flux))
    p /= fill
    f*=1e6
    return f,p

def create_timeseries(f, p):
	bw = f[1]-f[0]
	p = p*bw
	real_comp = np.random.normal(0, 1, len(p)) * np.sqrt(p/2.0)
	imag_comp = np.random.normal(0, 1, len(p)) * np.sqrt(p/2.0)

	inv_fft = np.zeros(len(p), dtype=complex)
	for i in range(len(inv_fft)):
		inv_fft[i] = real_comp[i] + 1j*imag_comp[i]

	x = np.fft.irfft(inv_fft) * len(p)
	return x

def normalise_ts(freq, power, time, flux):
    lhs=(1./float(len(time)))*np.sum(flux**2) 
    bw = freq[1]-freq[0]
    # Power in ppm^2uHz^-1 -> needs to be in ppm^2bin^-1 for parseval!
    rhs= np.sum(power*bw)
    ratio=lhs/rhs

    # Divide by square root to put ratio into amplitude!!!!!
    flux = flux / np.sqrt(ratio)
    return flux

def create_window(t,w):
	tmp=np.ones(len(t))
	tmp[(t/3600)%24<(24-w)]=0#8 hour observing windows not including random gaps
	return tmp

if __name__ == "__main__":
	#np.random.seed(143)
	home=expanduser('~')
	folder=os.getcwd()
	star='../RV/Artificial_Spectra/Subgiant_numax1000/Subgiant_ps.pow'
	fin, pin = np.loadtxt(star, unpack=True)
	x = create_timeseries(fin, pin)

	# Cadence of new observations
	bw=fin[1]-fin[0]
	dt = 1e6/(2*fin[-1])
	Tlen=1e6/bw# in s
	print('Cadence (s):',dt)
	#Length of observations of RV-days
	T=200

	# Create time array
	time = np.arange(0, Tlen, dt)#in seconds
	x = normalise_ts(fin, pin, time, x)
	x=x[(time/86400)<T]
	time=time[(time/86400)<T]

	#Window length-observations for w hours per day
	w=8

	# Generate power spectrum on truncated timeseries
	freq, power = ps(time,x)
	
	#np.savetxt(mypath+'/gapless_ps.pow',np.c_[freq,power])#,fmt='%10.4f')
	# Compare power spectra
	plt.plot(fin, pin, 'k',alpha=0.6,label='Original')
	plt.plot(freq, power, 'r', alpha=0.5,label='PS-$>$TS-$>$PS')
	plt.legend(loc='best')
	plt.xscale('log')
	plt.yscale('log')
	plt.show()

	# Create window function
	window= create_window(time,w)#8 hour observations Gives 33% fill

	#Additional Gaps on top of top hat function
	#idx=np.where(window==1)[0]
	#ngaps=500
	#idx=np.random.choice(idx,ngaps)
	#durn = np.random.gamma(1, 15, ngaps).astype(int)
	#for j in range(len(idx)):
		#window[idx[j]:idx[j]+(durn[j])] = 0.0
	fill_fac=len(window[window==1])/len(window)
	
	#Apply Window
	x = x*window
	#plt.plot(time/86400,window)
	#plt.show()

	#plt.plot(time/86400, x,'k')#'r',alpha=0.5)
	#plt.xlabel(r'Time (days)', fontsize=18)
	#plt.ylabel(r'Velocity (ms$^{-1}$)', fontsize=18)
	#plt.show()

	# Generate power spectrum with window applied
	freq, power = ps(time,x)

	# Plot to compare input spectra to chopped down and gapped data
	plt.plot(fin, pin, 'k', label='Original')
	plt.plot(freq, power, 'r',label='butchered')
	plt.legend(loc='best')
	plt.xlabel('Frequency ($\mu$Hz)', fontsize=18)
	plt.ylabel(r'PSD (m$^{2}$s$^{-2}\mu$Hz$^{-1}$)', fontsize=18)
	plt.show()

	#Fill on data
	print('Fill (%):',100*(fill_fac))















