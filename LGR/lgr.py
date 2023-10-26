

#%%


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, Rbf
import numpy.linalg as la
from sklearn.neighbors import NearestNeighbors

from jacobian import setRegressionFunction


# Convert an arbitrary list of particles to a time-oriented data frame:
def generateDF(particleList, K) -> pd.DataFrame:
    # Generate a time-oriented data frame where each row represents a time step
    
    # For the KNN algorithm:
    K = K+1
    
    # Remove any particle trajectories of length 1
    particleList = [p for p in particleList if len(p.pos) > 1]
    
    # dimension of the flow
    d = np.size(particleList[0].pos[0])
    n_particles = len(particleList)
    
    # Create a data frame with the appropriate columns. 
    columnNames = ['time','inframe','indices','positions','KNN']
    df = pd.DataFrame(columns=columnNames)
    
    # Get a list of all times that particles are observed:
    times = np.array([])
    for p in particleList:
        # All unique times
        utimes = np.unique(times)
        # get the times of particle i
        time = p.t
        # compare with all unique times and add any new times to time list.
        if not np.setxor1d(utimes, time).size == 0:
            diff = np.setdiff1d(time, utimes)
            times = np.append(times, diff)
    np.sort(utimes)
            
    # Iterate through particles to get the particle indices, positions, and K nearest neighbors at each time step
    # Assumes that particles will have continuous trajectories (ie. all times filled between t[0] and t[-1])
    for c, t in enumerate(utimes-1):
        if c % 25 == 0:
            print(f'KNN: t = {t}')
        inframe = []
        indices = []
        positions = []
        for i,p in enumerate(particleList):
            indices.append(i)
            # If the particle is in the domain at time t and at time t+1
            if t >= p.t[0] and t <= p.t[-1] and utimes[c+1] >= p.t[0] and utimes[c+1] <= p.t[-1]:
                inframe.append(1)
                tidx = np.argmin(np.abs(p.t - t))   # What is the index of the particle trajectory at time t?
                pos_at_t = p.pos[tidx,:]            # Position at time t for particle p
                positions.append(pos_at_t)
            else:
                inframe.append(0)
                positions.append(np.nan * np.ones(d))
        
        inframe = np.array(inframe)         # Particles that are in the frame at this time
        positions = np.array(positions)     # Positions of those particles
        indices = np.array(indices)         # The global indices of those particles
        
        # Get valid positions and indices at the given time for KNN purposes
        valid_pos = positions[np.array(inframe)==1]
        valid_idx = indices[np.array(inframe)==1]
        
        # Get the k nearest neighbors
        knn = np.nan * np.ones((n_particles, K))
        neigh = NearestNeighbors(n_neighbors=K).fit(valid_pos)
        knn_indices = neigh.kneighbors(valid_pos, return_distance=False)
        for i, v in enumerate(valid_idx):
            global_knn_indices = np.array([valid_idx[j] for j in knn_indices[i,:].tolist()]) # indices assoc. with global particle map
            knn[v,:] = np.copy(global_knn_indices)

        # Update thge dataframe
        new_row = pd.Series({
            'time' : t,
            'inframe' : inframe,
            'indices' : indices,
            'positions' : positions,
            'KNN' : knn
        })
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)   

    return df


# Calculate the Flow map Jacobian and Velocity Gradient at each time step:
def calcJacobianAndVelGrad(df, regfun=None) -> None:
    
    # Get number of time steps
    n_steps = len(df.index)
    n_particles = len(df.loc[0]['inframe'])
    
    # dimension of the data
    d = np.size(df.positions[0][0])
    
    # regression function
    def standard_regression(X0,X1):
        return X1 @ np.linalg.pinv(X0)
    if regfun is None:
        regfun = standard_regression
    
    # Add new columns to the data frame
    df["relF"] = None       # Relative Deformation Gradient (jacobian)
    df["L"] = None          # Velocity Gradient
    
    # Step through all times in the data set.  
    c=0
    for t in range(n_steps-1):
        if c % 25 == 0:
            tnow = df.loc[t,'time']
            print(f'Regression: t = {tnow}')
        c+=1
        # Specify the initial and final times
        df0 = df.iloc[t];  df1 = df.iloc[t+1]
        dt = df1['time'] - df0['time']
        
        # keep overlapping indices only.
        valid_indices = df0.inframe   
        
        # Allocate array of Jacobians for particle i
        Jacobian = np.nan * np.ones((d,d,n_particles))
        L = np.copy(Jacobian)
        
        # Iterate through the particles in frame at time t and t1
        for i, valid in enumerate(valid_indices):
            
            if valid:    # Particle persists to time 2.  Do computations
                X0 = [];  X1 = []
                O0 = df0.positions[:][i]        # Origin particle at time 0
                O1 = df1.positions[:][i]        # Origin particle at time 1
                
                # If all neighbors do not exist at second time
                if all(not valid_indices[j] for j in np.int64(df0.KNN[i][1:])):
                    break
                
                # Compute X matrices
                for j in np.int64(df0.KNN[i][1:]): 
                    if valid_indices[j]:
                        N0 = df0.positions[:][j]    # Neighbor particle at time 0    
                        N1 = df1.positions[:][j]    # Neighbor particle at time 1
                        X0.append(N0-O0)
                        X1.append(N1-O1)
                X0 = np.array(X0).T     # Format to be [d-by-k]
                X1 = np.array(X1).T

                # regress the function and store the gradient
                newF = regfun(X0,X1)
                if np.isnan(newF[0,0]):
                    raise ValueError
                newL = (newF - np.eye(d))/dt
                
                # Store
                Jacobian[:,:,i] = newF
                L[:,:,i] = newL

        # Store
        df['relF'][t] = Jacobian
        df['L'][t] = L


def computeMetrics(df, Tmax, metric_list='all'): 
    
    # dimension of the data
    d = np.size(df.positions[0][0])
    
    # number of particles
    n_particles = np.size(df.indices[0])
    
    def CalcC(df, idx0, idx1):
        
        # try:
        # Objective calculation of C from compositions
        C_list = np.nan * np.ones((d,d, n_particles))
        
        # Compute valid particle indices
        valid = 1-np.array(np.isnan(df['relF'][idx0] * df['relF'][idx1-1]))[0,0]
        # valid = np.array(df['inframe'][idx0] * df['inframe'][idx1])

        valid_indices = df.indices[0][valid==1]
        
        for j in valid_indices:    # Loop the particles that exist the entire time
            
            A = np.eye(d)
            for t in range(idx0, idx1):  # Loop the time indices
                A = df.loc[t,'relF'][:,:,j] @ A
                
            C = A.T @ A
            C_list[:,:,j] = C
            
            
        df['C'][idx0] = C_list
        # except:
        #     # raise ValueError('Error computing C')
        #     return
                
    def FTLE(df, idx0, idx1, T):
        
        # try:
        ftle_list = np.nan * np.ones((n_particles,1))
        
        # Compute valid indices
        valid = 1-np.array(np.isnan(df['relF'][idx0] * df['relF'][idx1-1]))[0,0]
        # valid = np.array(df['inframe'][idx0] * df['inframe'][idx1])
        valid_indices = df.indices[0][valid==1]
        
        for i in valid_indices:
            C = df.loc[idx0,'C'][:,:,i]
            lam, _ = la.eig(C)     
            ftle = 1./np.abs(T)*np.log(np.sqrt(np.max(lam)))
            ftle_list[i] = ftle
        df['ftle'][idx0] = ftle_list
        # except:
        #     return
            
    def LAVD(df, idx0, idx1):
        
        try:  
            lavd_list = np.nan * np.ones((n_particles,1))
            
            # Compute valid indices (assumes continuous if true)
            valid = 1-np.array(np.isnan(df['relF'][idx0] * df['relF'][idx1-1]))[0,0]

            valid_indices = df.indices[0][valid==1]
            for i in range(idx0, idx1):
                avg_vort = np.nanmean(df.loc[i,'vorticity'])
                dt = df.loc[i+1,'time'] - df.loc[i,'time']
                for j in valid_indices:
                    lavd_list[j] = np.nansum((lavd_list[j],np.abs(df.loc[i,'vorticity'][j]-avg_vort)*dt))
            
            df['lavd'][idx0] = lavd_list
        except:
            return
    
    def DRA(df, idx0, idx1):
        
        # only valid for 2d flows
        assert len(df['positions'][0]) == 2, "DRA cannot be computed for 2d flows in this implementation,"
        
        try:  
            dra_list = np.nan * np.ones((n_particles,1))
            
            # Compute valid indices (assumes continuous if true)
            valid = 1-np.array(np.isnan(df['relF'][idx0] * df['relF'][idx1-1]))[0,0]
            
            valid_indices = df.indices[0][valid==1]
            for i in range(idx0, idx1):
                dt = df.loc[i+1,'time'] - df.loc[i,'time']
                for j in valid_indices:
                    dra_list[j] = np.nansum((dra_list[j], df.loc[i,'vorticity'][j] * dt))
            
            df['dra'][idx0] = dra_list
        except:
            return
    
    # Number of total time intervals
    n_steps = len(df.index)
    
    # Specify the metrics being computed
    if metric_list == 'all':
        metric_list = ['ftle', 'lavd', 'dra', 'vort']
    
    # Set columns in the dataframe
    if 'ftle' in metric_list:
        df['C'] = None
        df['ftle'] = None
    if 'lavd' in metric_list:
        df['lavd'] = None
    if 'dra' in metric_list:
        df['dra'] = None
    
    # At every time step, iterate through the particles and compute the metric fields.
    c=0
    for i in range(n_steps-1):
        
        if c % 25 == 0:
            tnow = df.loc[i,'time']
            print(f'Metrics: t = {tnow}')
        c+=1      
        
        # For the Lagrangian metrics (over domain Tmax)
        T = 0   # time of Lagrangian computations
        j = 0   # second index
        while T <= Tmax + 0.001:
            if i + j + 1 > n_steps-1:
                T = Tmax + 100      # break the loop
            else:
                dt = df['time'][i+j+1] - df['time'][i+j] 
                j += 1;  T += dt
                
        if T >= Tmax:
            if 'ftle' in metric_list:
                CalcC(df,i,i+j)
                FTLE(df,i,i+j,T)
            if 'lavd' in metric_list:
                LAVD(df,i,i+j)
            if 'tor' in metric_list:
                DRA(df,i,i+j)

def interpolate_2D(scatteredData, gridvectors, mask=None, method='cubic'):
    # scatteredData should be a list with a tuple containing (x, y) and the values z
    
    # First, make the target grid to interpolate to.  
    xvec = gridvectors[0]
    yvec = gridvectors[1]
    X, Y = np.meshgrid(xvec, yvec)
    
    # Interpolate the data:
    Z = griddata(scatteredData[0], scatteredData[1], (X, Y), method=method)
    
    # masking
    if not mask is None:
        Z[mask] = np.nan
    
    return Z

def rbf_interp(scatteredData, gridvectors, mask=None, method='multiquadric'):
    
    # First, make the target grid to interpolate to.  
    xvec = gridvectors[0]
    yvec = gridvectors[1]
    X, Y = np.meshgrid(xvec, yvec)
    xtest = X.reshape([np.size(X)])
    ytest = Y.reshape([np.size(Y)])
    
    # Get the scattered data in the right format.
    xdata = scatteredData[0][0]
    ydata = scatteredData[0][1]
    zdata = scatteredData[1]
    
    # Interpolate the data:
    rbf_fun = Rbf(xdata, ydata, zdata, function=method)
    z = rbf_fun(xtest, ytest)
    Z = z.reshape(np.shape(X))
    
    # masking
    if not mask is None:
        Z[mask] = np.nan
    
    return Z

def generateFields(df, gridvectors, metric_list='all', approach='interp', method='cubic', mask=None, **kwargs):
    # 2D only
    
    def get_valid_indices(positions, data):
        
        indices = [i for i in range(len(data)) if not np.isnan(data)[i]]
        data_out = data[indices]
        x_out = positions[0][indices]
        y_out = positions[1][indices]
        
        return (x_out, y_out), data_out
    
    # number of time steps
    n_steps = len(df.index)
    
    # Specify the metrics being computed
    if metric_list == 'all':
        metric_list = ['ftle', 'lavd', 'dra', 'vort']
        
    # data frame column for scalar fields
    df["ScalarFields"] = None
    
    fieldsList = []
    c=0
    for t in range(n_steps-1):  # iterate through times
        
        if c % 25 == 0:
            tnow = df.loc[t,'time']
            print(f'Interpolation: t = {tnow}')
        c+=1
        
        dft = df.loc[t]
        
        # Format position data to a tuple
        pos = dft['positions']
        x = pos[:,0]
        y = pos[:,1]
        positions = (x, y)
        
        # Get all of the data that is desired and store it in a dictionary:
        scalarFields = dict()
        for s in metric_list:
            
            if s == 'ftle':
                data = dft['ftle']
                
            if s == 'lavd':
                data = dft['lavd']
                
            if s == 'dra':
                data = dft['dra']
                
            if s == 'vort':
                data = dft['vorticity']
            
            if not data is None:
                try:
                    if all(item == 0.0 for item in data) or all(item == np.nan for item in data):
                        scalarFields[s] = None
                    else:
                        scatteredData = get_valid_indices(positions, data)
                        if approach == 'interp':
                            field = interpolate_2D(scatteredData, gridvectors, mask=mask, method=method)
                        elif approach == 'rbf':
                            field = rbf_interp(scatteredData, gridvectors, mask=None, method=method)
                        scalarFields[s] = field
                except:
                    # raise ValueError('interpolation error')
                    continue
        
        # Data managment
        df['ScalarFields'][t] = scalarFields
        fieldsList.append(scalarFields)
    
    # output value if specified
    if 'outputfields' in kwargs:
        if kwargs['outputfields']:  
            return scalarFields
        







#%%
if __name__ == "__main__":
    
    load_comp_data = False
    
    if load_comp_data:
        comp_Data_file = '/home/tannerharms/TannerHarms/data/ParticleMaps/DoubleGyre/Structured/PM_DoubleGyre_Structured_300.0particles_t00_T20_dt0.1.pkl'
        df = pd.read_pickle(comp_Data_file)
        
        nx = 100
        xvec = np.linspace(-0.0098, 0.0798,nx)
        yvec = np.linspace(-0.0198,0.0198,int(nx/2.5))
        gridvectors = [xvec, yvec]
        # generateFields(df, gridvectors, approach='interp', method='cubic')
    else:
        # Load in a data set
        datafilein = r'/home/tannerharms/TannerHarms/data/ShearLayerDataSeets/ParticleMaps/ShearLayerPTV_ParticleTracks.pkl'
        # datafilein = '/home/tannerharms/TannerHarms/data/ParticleMaps/DoubleGyre/Random/PM_DoubleGyre_Rand_100particles_t00_T25_dt0.1.pkl'
        particleList = pd.read_pickle(datafilein)
        
        # save the data
        sv = True
        # svfldr = '/home/tannerharms/TannerHarms/data/ParticleMaps/DoubleGyre'
        # svname = f'PM_DoubleGyre_tik_test.pkl'
        # svpath = os.path.join(svfldr,'MetComputed',svname)
        svpath = r'/home/tannerharms/TannerHarms/data/ShearLayerDataSeets/ParticleMaps/ShearLayerPTV_MetComputed.pkl'
        
        df = generateDF(particleList,45)
        df.info(memory_usage="deep")
        
        # set a regression function
        # regfun = setRegressionFunction( kernel='kernelRegression', kernel_fun='ratquad', params=None,#
                                        # regularize=True, lam=0.00001)# kernel='radialGaussian' )#
        regfun = setRegressionFunction()
        # regfun = setRegressionFunction(kernel='kernelRegression', kernel_fun='rbf', params=None,#
        #                                 regularize=True, lam=0.00000001)# regularize for stability
        # regfun = setRegressionFunction(kernel='radialGaussian',regularize=True, lam=0.0000000001)
        calcJacobianAndVelGrad(df, regfun=regfun)
        
        # save the data
        df.to_pickle(svpath)
        
        computeMetrics(df, 5/190)
        
        # datafilein = '/home/tannerharms/TannerHarms/data/ParticleMaps/DoubleGyre/MetComputed/PM_DoubleGyre_500Particles_test_computed_nofields.pkl'
        # df = pd.read_pickle(datafilein)
        
        # save the data
        df.to_pickle(svpath)
        
        m2pix = 10**(-5)
        
        nx = 150
        xvec = np.linspace(0, m2pix*(1920), nx)
        yvec = np.linspace(-1200*m2pix/2,1200*m2pix/2,int(round(nx*1200/1920)))
        # xvec = np.linspace(-0.0098, 0.0798,nx)
        # yvec = np.linspace(-0.0198,0.0198,int(nx/2.5))
        gridvectors = [xvec, yvec]
        generateFields(df, gridvectors, approach='interp', method='cubic')#, method='cubic')
        
        # save the data
        df.to_pickle(svpath)
    
    print(df.head())
    df.info(memory_usage="deep")
    
    #%%
    # nx = 100
    # xvec = np.linspace(0,2,nx)
    # yvec = np.linspace(0,1,int(nx/2))
    
    X, Y = np.meshgrid(xvec, yvec)
    xlim = [np.min(xvec), np.max(xvec)]
    ylim = [np.min(yvec), np.max(yvec)]
    
    tstep = 8

    x = df.loc[tstep,'positions'][:,0]
    y = df.loc[tstep,'positions'][:,1]

    ftle = np.squeeze(df.loc[tstep,'ScalarFields']['ftle'])
    lavd = np.squeeze(df.loc[tstep,'ScalarFields']['lavd'])
    ftQ = np.squeeze(df.loc[tstep,'ScalarFields']['ftQ'])
    vort = np.squeeze(df.loc[tstep,'ScalarFields']['vort'])

    fig, axs = plt.subplots(2,2,sharex=True, sharey=True, figsize=[12,6])
    plt.subplots_adjust(hspace = 0.05, wspace = 0.15)

    clim = [0, np.nanmax(ftle)]
    ftleim = axs[0,0].pcolormesh(X, Y, ftle, cmap='gray2hot', vmin=clim[0], vmax=clim[1])
    axs[0,0].scatter(x,y, s=2, c=[[0.6, 0.6, 0.6, 0.7]])
    axs[0,0].axis('scaled')
    axs[0,0].set_xlim(xlim)
    axs[0,0].set_ylim(ylim)
    divider = make_axes_locatable(axs[0,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(ftleim, cax=cax)
    # axs[0,0].set_title('FTLE')

    clim = [0, np.nanmax(lavd)]
    lavdim = axs[0,1].pcolormesh(X, Y, lavd, cmap='viridis', vmin=clim[0], vmax=clim[1])
    axs[0,1].scatter(x,y, s=2, c=[[0.6, 0.6, 0.6, 0.7]])
    axs[0,1].axis('scaled')
    axs[0,1].set_xlim(xlim)
    axs[0,1].set_ylim(ylim)
    divider = make_axes_locatable(axs[0,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(lavdim, cax=cax)
    # axs[0,1].set_title('LAVD')

    temp = ftQ
    cm = stitchColormaps(temp, 'pink_r','bone', 0)
    clim = [np.nanmin(temp), np.nanmax(temp)]
    torim = axs[1,0].pcolormesh(X, Y, temp, cmap=cm, vmin=clim[0], vmax=clim[1])
    axs[1,0].scatter(x,y, s=2, c=[[0.6, 0.6, 0.6, 0.7]])
    axs[1,0].axis('scaled')
    axs[1,0].set_xlim(xlim)
    axs[1,0].set_ylim(ylim)
    divider = make_axes_locatable(axs[1,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(torim, cax=cax)
    # axs[1,0].set_title('TOR')

    clim = [-np.abs(np.nanmax(vort)), np.abs(np.nanmax(vort))]
    clim = [-25, 25]
    vortim = axs[1,1].pcolormesh(X, Y, vort, cmap='bwr', vmin=clim[0], vmax=clim[1])
    axs[1,1].scatter(x,y, s=2, c=[[0.6, 0.6, 0.6, 0.7]])
    axs[1,1].axis('scaled')
    axs[1,1].set_xlim(xlim)
    axs[1,1].set_ylim(ylim)
    divider = make_axes_locatable(axs[1,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(vortim, cax=cax)
    # axs[1,1].set_title('Vorticity')

    fig.savefig('test', dpi=450)
    
    
# %%
