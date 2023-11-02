
#%%
import sys, copy, os
import numpy as np

from LGR.lgr import *
from LGR.jacobian import *
from LGR.classes import *
from LGR.plotting import *
from Flows.Flows import *

# Create a class that marks a planet-satellite type particle.
# For this example, we will just implement finite-differences for gradient computation.
class PSParticle:
    
    def __init__(self, flow, initial_position, times, dx, domain=None, resample=False): 
        
        # Basic properties
        self.flow = flow                # the flow function for generating trajectories
        self.pos0 = initial_position    # initial condition of the planet particle
        self.t = times                  # vector of times for computation.
        self.dx = dx                    # spacing to satellite particles
        self.dim = len(self.pos0)       # dimensionality of the flow

        # In case the boundary of the flow is important
        if domain is None:              # boundaries that the flow is constrained to
            self.domain = [[-np.inf, np.inf] for i in range(len(self.pos0))]
        else:
            assert(len(domain) == len(self.pos0))
            self.domain = domain
        
        # choose behavior based on whether resampling is used
        if resample:
            # Initialize variables
            self.pos = np.zeros((len(self.t), self.dim))
            self.pos[0,:] = np.copy(self.pos0)
            self.satpos0 = np.zeros((2*self.dim, len(self.t), self.dim))
            self.satpos1 = np.zeros((2*self.dim, len(self.t), self.dim))
            
            # Iterate through the times along the trajectory
            A = np.eye(self.dim)
            for i in range(len(self.t)-1):
                # sample fresh satellite ICs on a grid.
                # print(self.pos[i,:])
                satICs = self.sampleICs(self.pos[i,:])
                ICs = np.append(self.pos[i,:][np.newaxis], satICs, axis=0)
                # print(satICs, self.pos[i,:])
                
                # establish the time vector for the timestep
                tvec = [self.t[i], self.t[i+1]]
                
                # Compute trajectories
                trajectories = self.computeTrajectories(ICs, tvec)
                
                # Organize into data matrices and compute short-time jacobian
                X1 = trajectories[1:,-1,:] - trajectories[0,-1,:]
                A_inst = self.computeJacobian(X1)
                
                # Apply composition
                A = A_inst @ A
                
                # Update positions
                self.pos[i+1,:] = trajectories[0,-1,:].squeeze()
                self.satpos0[:,i,:] = trajectories[1:,0,:]
                self.satpos1[:,i,:] = trajectories[1:,-1,:]
            
            self.jacobian = A
            self.ftle = self.computeFTLE()  
            
        else:
            # samplie initial conditions
            satICs = self.sampleICs(self.pos0)
            ICs = np.append(self.pos0[np.newaxis], satICs, axis=0)
            
            # Compute the trajectories
            trajectories = self.computeTrajectories(ICs, self.t)
            self.pos = trajectories[0,:,:].squeeze()
            self.satpos = trajectories[1:,:,:]
            
            # Organize into data matrices and compute metrics
            X1 = trajectories[1:,-1,:] - trajectories[0,-1,:]
            self.jacobian = self.computeJacobian(X1)
            self.ftle = self.computeFTLE()

    # Get initial conditions for the satellite particles using finite-difference sampling
    def sampleICs(self, planet_pos):
        satICs = np.nan * np.ones((2*self.dim, self.dim))
        for d in range(self.dim):
            perturbation = np.zeros_like(planet_pos)
            perturbation[d] = self.dx
            
            # get the plus and minus perturbations
            pos_minus = planet_pos - perturbation
            pos_plus = planet_pos + perturbation
            
            # remove any ICs that are outside of the boundary
            if pos_minus[d] < self.domain[d][0] and pos_plus[d] > self.domain[d][1]:
                raise ValueError('dx value is too large for the given domain.')
            elif pos_minus[d] < self.domain[d][0]:     # If the lower perturbation is OOB
                satICs[2*d+1,:] = pos_plus
            elif pos_plus[d] > self.domain[d][1]:      # If the upper perturbation is OOB
                satICs[2*d,:] = pos_minus
            else:                                   # If no perturbation is OOB
                satICs[2*d,:] = pos_minus
                satICs[2*d+1,:] = pos_plus
            
        return satICs
    
    # Compute trajectories for the seeded particles.
    def computeTrajectories(self, ICs, tvec):
        # Initialize
        trajectories = np.nan * np.ones((1+2*self.dim, len(tvec), self.dim))
        
        # Find valid particles for integration
        idxs = [i for i in range(1+2*self.dim) if not np.isnan(ICs[i,0])]
        ICs = np.array([ICs[i,:] for i in range(1+2*self.dim) if not np.isnan(ICs[i,0])])
        
        # compute valid trajectories
        self.flow.initial_conditions = np.copy(ICs)
        self.flow.time_vector = np.copy(tvec)
        self.flow.integrate_trajectories()
        computed = self.flow.states
        
        # reinsert to the trajectories list
        for c, i in enumerate(idxs):
            trajectories[i,:,:] = computed[c,:,:]
        
        self.flow.initial_conditions = None
        self.flow.time_vector = None
        self.flow.states = None
        
        return trajectories
    
    # use finite-differences to compute the jacobian  
    def computeJacobian(self, X1):
           
        # Initialize:
        grad = []
        
        # Iterate through the dimensions
        c = 0
        for i in range(self.dim):
            c = 2*i
            # the deformation in dimension i
            DimVec = np.array([X1[c,:], np.zeros([self.dim]), X1[c+1,:]])
            
            # Manage NaNs.  
            # If only one nan exists, compute the forward/backward finite difference.
            # If no NaNs, then compute central differences.
            if np.sum(np.isnan(DimVec)) >= 2*self.dim:
                grad.append(np.nan*np.ones_like(DimVec))
            elif np.sum(np.isnan(DimVec)) == 1*self.dim:
                diff = np.diff(DimVec, axis=0)/self.dx
                diff = diff[~np.isnan(diff)]
                DimVec[1,:] = diff
                grad.append(DimVec)
            else:
                grad.append(np.gradient(DimVec,self.dx,axis=0))

        # grad should have [[dxdx dxdy ...][dydx dydy ...]...]
        assert np.shape(grad) == (self.dim,3,self.dim)
        
        # Assemble the jacobian from grad
        A = np.zeros([self.dim, self.dim])
        for i in range(self.dim):
            for j in range(self.dim):                    
                # Take the middle element of each of the grad vectors from above.
                A[i,j] = grad[i][1,j]
                
        return A
    
    def computeFTLE(self):
        U, S, VT = np.linalg.svd(self.jacobian)
        return 1/np.abs(self.t[-1]-self.t[0])*np.log(S[0])
        
        
if __name__ == "__main__":
    
    # Generate structured trajectory data for the double gyre flow

    # Flow type and duration:
    flowname = "Gyre"
    n_steps = int(15/0.25)
    dt = 0.25    

    # optional parameters
    parameters = {  # These are specified as defaults as well. 
        "A": 0.1,
        "epsilon": 0.1,
        "omega":2*np.pi/10
    }

    # Initialize the flow 
    flow = Flow()
    flow.predefined_function(flowname)
    flow_c = copy.deepcopy(flow)

    # Now, make vectors associated with each axis.
    domain = np.array([[0, 2],[0, 1]])
    n_y = 50            # number of rows
    n_x = 2*n_y         # number of columns
    eps = 0.1001        # for visualization
    y_vec = np.linspace(domain[1,0]+eps, domain[1,1]-eps, n_y+1)     # 25 rows
    x_vec = np.linspace(domain[0,0]+eps, domain[0,1]-eps, n_x+1)     # 50 columns

    # Then, make the mesh grid and flatten it to get a single vector of positions.  
    mesh = np.meshgrid(x_vec, y_vec)
    # x = mesh[0].reshape(-1,1)
    # y = mesh[1].reshape(-1,1)
    # ICs = np.append(x, y, axis=1)
    
    # Generate a time vector
    tvec = np.linspace(0, dt*n_steps, n_steps+1) 
    
    # Create PS particles for jacobian computations.
    dx = 0.1
    ftle_field_resampled = np.zeros((len(y_vec), len(x_vec)))
    ftle_field_traditional = np.zeros((len(y_vec), len(x_vec)))
    c=0
    pvals = []
    for i, y in enumerate(y_vec):
        for j, x in enumerate(x_vec):
            
            if c % 25 == 0:
                print(f"particle {c} out of {len(y_vec) * len(x_vec)}")
            
            ic = np.array([x,y])
            
            # compute FTLE without resampling
            p_traditional = PSParticle(flow_c, np.copy(ic), tvec, dx, domain=domain, resample=False)
            ftle_field_traditional[i,j] = p_traditional.ftle
            
            # compute FTLE with resampling
            p_resampled = PSParticle(flow, np.copy(ic), tvec, dx, domain=domain, resample=True)
            ftle_field_resampled[i,j] = p_resampled.ftle
            c+=1 
        

    # Plot an example to see if it is working
    #%%
    
    fig, axs = plt.subplots(1,2)
    
    clim = [0,0.5]
    
    ftleim = axs[1].pcolormesh(mesh[0], mesh[1], ftle_field_resampled, cmap="gray2hot", vmin=clim[0], vmax=clim[1])
    axs[1].axis('scaled')
    axs[1].set_title('FTLE with resampling')
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(ftleim, cax=cax)
    
    axs[0].pcolormesh(mesh[0], mesh[1], ftle_field_traditional, cmap="gray2hot", vmin=clim[0], vmax=clim[1])
    axs[0].axis('scaled')
    axs[0].set_title('FTLE without resampling')
    
    plt.show()
    
    #%%