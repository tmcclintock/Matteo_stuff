"""
The model for Matteo's mass function
"""
import numpy as np
import cosmocalc as cc
from scipy import integrate

mass_pivot=6.4929581370445531e13
volume = 1050.0**3 #[Mpc/h]^3; simulation volume

def set_cosmology(cos):
    ombh2,omch2,w0,ns,ln10As,H0,Neff,sigma8 = cos
    h = H0/100.
    Ob = ombh2/h**2
    Om = Ob + omch2/h**2
    cosmo_dict = {"om":Om,"ob":Ob,"ol":1-Om,"ok":0.0,"h":h,"s8":sigma8,"ns":ns,"w0":w0,"wa":0.0}
    cc.set_cosmology(cosmo_dict)
    return

def sq_model(lM,s,q,a):
    M = np.exp(lM)
    return cc.tinker2008_mass_function(M,a,200)*(s*np.log10(M/mass_pivot)+q)

def N_in_bin(lM_bins,s,q,a):
    lM_bins = np.log(10**lM_bins)#switch to natural log
    return volume * np.array([integrate.quad(sq_model,lMlow,lMhigh,args=(s,q,a),epsabs=TOL,epsrel=TOL/10.)[0] for lMlow,lMhigh in lM_bins])
