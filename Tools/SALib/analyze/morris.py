from __future__ import division
from ..util import read_param_file
from sys import exit
import numpy as np

# Perform Morris Analysis on file of model results
def analyze(pfile, input_file, output_file, column = 0, delim = ' '):
    
    param_file = read_param_file(pfile)
    Y = np.loadtxt(output_file, delimiter = delim)
    X = np.loadtxt(input_file, delimiter = delim)
    
    if Y.ndim > 1:
        Y = Y[:, column]
    
    D = param_file['num_vars']
    
    if Y.size % (D+1) == 0:    
        N = int(Y.size / (D + 1))
    else:
        print """
                Error: Number of samples in model output file must be a multiple of (D+1), 
                where D is the number of parameters in your parameter file.
              """
        exit()            
    
    ee = np.empty([N, D])
    
    # For each of the N trajectories
    for i in range(N):
        
        # Set up the indices corresponding to this trajectory
        j = np.arange(D+1) + i*(D + 1)
        j1 = j[0:D]
        j2 = j[1:D+1]
        
        # The elementary effect is (change in output)/(change in input)
        # Each parameter has one EE per trajectory, because it is only changed once in each trajectory
        ee[i,:] = np.linalg.solve((X[j2,:] - X[j1,:]), Y[j2] - Y[j1]) 
    
    # Output the Mu, Mu*, and Sigma Values
    print "Parameter Mu Sigma Mu_Star"
    for j in range(D):
        mu = np.average(ee[:,j])
        mu_star = np.average(np.abs(ee[:,j]))
        sigma = np.std(ee[:,j])
        # mu_star_conf = compute_mu_star_confidence(ee[:,j], N, num_resamples)
        
        print "%s %f %f %f" % (param_file['names'][j], mu, sigma, mu_star)
        

#def compute_mu_star_confidence(ee, N, num_resamples):
#    
#    ee_resampled = np.empty([N])
#    mu_star_resampled  = np.empty([num_resamples])
#    
#    for i in range(num_resamples):
#        for j in range(N):
#            
#            index = np.random.randint(0, N)
#            ee_resampled[j] = ee[index]
#        
#        mu_star_resampled[i] = np.average(np.abs(ee_resampled))
#    
#    return 1.96 * mu_star_resampled.std(ddof=1)
    
        