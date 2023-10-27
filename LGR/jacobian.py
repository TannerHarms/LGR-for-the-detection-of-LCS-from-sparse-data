#  functions for computing the flow-map jacobian as in Harms et al. (2023)

import sys, os, copy
import numpy as np


# Generate a callable regression function for use in LGR and in planet-satellite approaches.
def setRegressionFunction(kernel:str=None, lam:float=None, sig:float=None ):

    # Specify regularizer
    if lam is None:
        lam = 0
    
    # Kernel functions to be used below
    gen_phi = lambda r, sigma: np.exp(-(r**2)/(2*sigma**2))
    
    # Make a function using X0 and X1. This is what will be returned by the parent function.
    def regFun(X0, X1):
        
        # naming variables
        dims, M = np.shape(X0)
        I = np.eye(dims)                

        # Set the kernel for the regression.
        K = np.eye(M)
        if kernel == 'radialGaussian':
            if sig is None: 
                # Compute kernel standard deviation from data
                sigma = np.std(np.linalg.norm(X0, axis=0))
                if sigma == 0:
                    sigma = 1
            else:
                sigma = sig
            # Radial Gaussian Weighting
            Phi = np.diag(gen_phi(np.linalg.norm(X0, axis=0), sigma)+0.00001)
            K = Phi.T * Phi

        # Return the function
        if not lam is None:
            return (X1 @ K @ X0.T) @ np.linalg.inv(X0 @ K @ X0.T + lam*M*I)
        else:   
            if kernel is None:
                return X1 @ np.linalg.pinv(X0)
            else:
                return (X1 @ K @ X0.T) @ np.linalg.inv(X0 @ K @ X0.T)
            
    return regFun



def jacobianFromVelocityGradient(particleMap, tvec, UpdateFunction, UpdateFunctionGradient):
    # the update function gradient should be v
    
    pm = particleMap
    
    # Get the initial particle states and time
    states, t0, _ = pm.getParticleStates()
    
    # Check that the integration starts at the right time.
    if tvec[0] != t0:
        tvec += t0
        
    for i in range(len(tvec)-1):
        print(f't = {tvec[i]} out of {tvec[-1]}')   
        
        t0 = tvec[i]
        t1 = tvec[i+1]
        
        #Iterate through all of the particles:
        for p in pm.particles:
            
            # Get the particle position, time, and time difference
            pos = p.pos
            t = p.t
            dt = t1-t0
            p.dt = dt
            
            # Get the gradient at that postion     
            gradv = UpdateFunctionGradient(pos, t)
            A_update = np.array(gradv * dt + np.eye(len(pos)))
            
            # Update the Jacbian on the particle.  
            p.A_inst = A_update
            p.A = A_update @ p.A
            p.L = gradv
        
        # Update particle positions.
        newstates = analyticalUpdate(UpdateFunction, states, t0, t1)
        pm.updateParticleStates(newstates, t1)
        
        for p in pm.particles:
            # Make sure to do the metric computations.
            p.computeMetrics()
        
        states, _, _ = pm.getParticleStates()