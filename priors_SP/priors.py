"""
Calculate priors on S & P.
"""
import numpy as np
import os,sys
import model
import emcee, corner
import matplotlib.pyplot as plt
import scipy.optimize as op

base = "../../../all_MF_data/building_MF_data/"
database = base+"full_mf_data/Box%03d_full/Box%03d_full_Z%d.txt"
covbase  = base+"covariances/Box%03d_cov/Box%03d_cov_Z%d.txt"
cosmos = np.genfromtxt("building_cosmos.txt")
cosmos = np.delete(cosmos,0,1) #delete the boxnumber
N_cos = len(cosmos)-1 #Last one is broken...
N_z = 10
a_array = np.array([0.25,0.333333,0.5,0.540541,0.588235,0.645161,0.714286,0.8,0.909091,1.0]) #scale factors

guess = np.array([0.0,1.1])
N_walkers = 8
N_steps = 2000
N_burn = 400

single_test = False
best_result = False
do_mcmc = True
show_corner = False
save_results = True
FROM_SCRATCH = False

savebase = "./txtfiles/"
N_dim = 2 #2 parameters, s & q
if FROM_SCRATCH:
    best_params = np.zeros((N_cos,N_dim))
    mean_params = np.zeros((N_cos,N_dim))
    var_params = np.zeros((N_cos,N_dim))
else:
    best_params = np.loadtxt(savebase+"best_params.txt")
    mean_params = np.loadtxt(savebase+"mean_params.txt")
    var_params = np.loadtxt(savebase+"var_params.txt")


def lnprior(params):
    s,q = params
    #if q <= 0.01: return -np.inf
    return 0

def lnlike(params,lM_bins_array,N_array,icov_array,a_array):
    s,q = params
    LL = 0
    for i in xrange(0,N_z):
        lM_bins = lM_bins_array[i]
        N = N_array[i]
        icov = icov_array[i]
        a = a_array[i]
        N_model = model.N_in_bin(lM_bins,s,q,a)
        X = N-N_model
        LL += -0.5*np.dot(X,np.dot(icov,X)) # + constant
    return LL

def lnprob(params,lM_bins_array,N_array,icov_array,a_array):
    lp = lnprior(params)
    if not np.isfinite(lp): retun -np.inf
    return lp + lnlike(params,lM_bins_array,N_array,icov_array,a_array)

for i in xrange(0,N_cos):
    cos = cosmos[i]
    model.set_cosmology(cos)

    #Read in everything
    lM_bins_array = []
    N_array = []
    icov_array = []
    for j in xrange(0,N_z):
        data = np.genfromtxt(database%(i,i,j))
        lM_bins_array.append(data[:,:2])
        N_array.append(data[:,2])
        icov_array.append(np.linalg.inv(np.genfromtxt(covbase%(i,i,j))))
    
    if single_test: #Just a simple test to see if it works.
        print lnprob(guess,lM_bins_array,N_array,icov_array,a_array)
                        
    if best_result:
        """
        Find the point in parameter space with a maximum likelihood.
        """
        nll = lambda *args: -lnlike(*args)
        result = op.minimize(nll, guess, args=(lM_bins_array,N_array,icov_array,a_array),method='Powell')
        best_params[i] = result['x']
        print "Result for Box%03d"%i,result['x']

    if do_mcmc:
        start = best_params[i]
        pos = [start + 1e-3*np.random.randn(N_dim) for j in range(N_walkers)]
        sampler = emcee.EnsembleSampler(N_walkers, N_dim, lnprob, args=(lM_bins_array,N_array,icov_array,a_array))
        print "Starting mcmc on Box%03d"%i
        sampler.run_mcmc(pos,N_steps)
        chain = sampler.flatchain
        print "\tDone with Box%03d"%i
        np.savetxt(savebase+"chain_Box%03d.txt"%i,chain)
        print "\tchain saved for Box%03d"%i

    if show_corner:
        chain = np.loadtxt(savebase+"chain_Box%03d.txt"%i)
        fig = corner.corner(chain,labels=[r"$s$",r"$q$"])
        plt.show()

if save_results:
    np.savetxt(savebase+"best_params.txt",best_params)
    np.savetxt(savebase+"mean_params.txt",mean_params)
    np.savetxt(savebase+"var_params.txt",var_params)
