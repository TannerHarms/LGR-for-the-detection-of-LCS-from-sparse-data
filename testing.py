import sys, copy, os
import numpy as np

from LGR.lgr import *
from LGR.jacobian import *
from LGR.classes import *
from LGR.plotting import *
from Flows.Flows import *

# Generate random trajectory data for the double gyre flow

# basic parameters:
flowname = "Gyre"
n_particles = 100
n_steps = 10
dt = 0.1    

# optional parameters
parameters = {  # These are specified as defaults as well. 
    "A": 0.1,
    "epsilon": 0.1,
    "omega":2*np.pi/10
}

# Initialize the flow 
flow = Flow()

# Generate random initial conditions on domain [[0,2],[0,1]] -> [x, y]
ICs = np.random.rand(n_particles,2)
ICs[:,0] *= 2

# Generate a time vector
tvec = np.linspace(0, dt*n_steps, n_steps) 

# Generate the trajectories
flow.predefined_function(flowname, ICs, tvec, parameters=parameters)
flow.integrate_trajectories()
trajectories = flow.states

# Generate a list of particles
particleList = []
for i in range(n_particles):
    state = np.squeeze(trajectories[i,:,:])
    particleList.append(SimpleParticle(state, tvec, i))
    
# plot the trajectories
plot_trajectories(trajectories)

# Set computational parameters
t = 15              # time duration for LCS analysis
kNN = 15            # Number of nearest neighbors to find for each particle
reg_type = 'None'   # 'None' or 'radialGaussian'
sigma = None        # standard deviation if reg_type is radial_gaussian
lam = 0.000000001   # Regularizer for the regression 
nx = 150            # grid length in x for interpolation (plotting)

# Generate the regression function
regfun = setRegressionFunction(kernel=reg_type, lam=lam, sig=sigma)

# specify which metrics to compute
metrics = ["ftle", "lavd", "dra", "vort"]

# Generate a data frame
df = generateDF(particleList, kNN)
n_particles = len(df['indices'][0])

# Perform the regressions
calcJacobianAndVelGrad(df, regfun=regfun)

# Compute the metrics on each particle trajectory
computeMetrics(df, t, metric_list=metrics)

# Interpolate the metrics onto a grid assuming the double gyre flow.
xvec = np.linspace(0,2,nx)
yvec = np.linspace(0,1,int(nx/2))   # square cells on the double gyre flow
gridvectors = [xvec, yvec]
if n_particles < 1000:
    generateFields(df, gridvectors, approach='rbf', method='multiquadric')
    interpstr = 'rbf_mq'
else:
    generateFields(df, gridvectors, approach='interp', method='cubic')
    interpstr = 'int3'
    
# plot the results
plotAllMetrics(df, xvec, yvec, tstep=0)
