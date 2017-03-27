"""
Calculate priors on S & P.
"""
import numpy as np
import os,sys
import model
import emcee
import matplotlib.pyplot as plt

base = "~/Desktop/all_MF_data/building_MF_data/"
database = base+"full_mf_data/Box%03d_full/Box%03d_full_Z%d.txt"
covbase  = base+"covariances/Box%03d_cov/Box%03d_cov_Z%d.txt"

cosmos = np.genfromtxt("building_cosmos.txt")
cosmos = np.delete(cosmos,0,1) #delete the boxnumber

def lnprior(params):
    s,q = params
    if q <= 0.01: return -np.inf
    return 0


for i in xrange(0,1):#len(cosmos)):
    cos = cosmos[i]
    model.set_cosmology(cos)
    Masses = np.logspace(13,15,num=10)
    dndm = np.array([model.sq_model(np.log(m),0,1.0,1.0) for m in Masses])
    print Masses
    print dndm
    plt.loglog(Masses,dndm)
    plt.show()
