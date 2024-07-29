import sys
import os
import numpy as np
import pandas as pd
from scipy.interpolate import griddata, Rbf
import numpy.linalg as la
from sklearn.neighbors import NearestNeighbors


sys.path.append(os.path.join(os.getcwd(), "ComputingTrajectories", "LGR"))
sys.path.append(os.path.join(os.getcwd(), "LGR"))


# Convert an arbitrary list of particles to a time-oriented data frame:
def generateDF(particleList, K, t_range=None) -> pd.DataFrame:
    # Generate a time-oriented data frame where each row represents a time step

    # For the KNN algorithm:
    K = K+1

    # Remove any particle trajectories of length 1
    particleList = [p for p in particleList if len(p.pos) > 1]

    # dimension of the flow
    d = np.size(particleList[0].pos[0])
    n_particles = len(particleList)

    # Create a data frame with the appropriate columns.
    columnNames = ['time', 'inframe', 'valid', 'indices', 'positions', 'KNN']
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
    if t_range is not None:
        utimes = utimes[(utimes > t_range[0]) & (utimes < t_range[1])]

    # Iterate through particles to get the particle indices, positions, and
    # K nearest neighbors at each time step.  Assumes that particles will
    # have continuous trajectories (ie. all times filled between
    # t[0] and t[-1])
    for c, t in enumerate(utimes):
        if c % 25 == 0:
            print(f'KNN: t = {t}')
        inframe = []
        valid = []
        indices = []
        positions = []
        for i, p in enumerate(particleList):
            indices.append(i)
            # If the particle is in the domain at time t and at time t+1
            if t >= p.t[0] and t <= p.t[-1]:
                inframe.append(1)
                tidx = np.argmin(np.abs(p.t - t))   # What is the index at t?
                pos_at_t = p.pos[tidx, :]           # Position at time t for p.
                positions.append(pos_at_t)
                if not c == len(utimes)-1:
                    if utimes[c+1] >= p.t[0] and utimes[c+1] <= p.t[-1]:
                        valid.append(1)
                    else:
                        valid.append(0)
            else:
                inframe.append(0)
                valid.append(0)
                positions.append(np.nan * np.ones(d))

        inframe = np.array(inframe)         # Particles that are in the frame
        positions = np.array(positions)     # Positions of those particles
        valid = np.array(valid)             # valid for computing the jacobian
        indices = np.array(indices)         # Their global indices

        # If in the last row, set time and position characteristics.
        if c == len(utimes)-1:
            new_row = pd.Series({
                'time': t,
                'inframe': inframe,
                'valid': valid,
                'indices': indices,
                'positions': positions,
                'KNN': []
            })
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
            break

        # Get valid positions and indices at the given time for KNN purposes
        # This way only neighbors that persist from this time to the next are
        # used in computations
        valid_pos = positions[np.array(valid) == 1]
        valid_idx = indices[np.array(valid) == 1]

        # Get the k nearest neighbors
        knn = np.nan * np.ones((n_particles, K))
        try:
            neigh = NearestNeighbors(n_neighbors=K).fit(valid_pos)
            knn_indices = neigh.kneighbors(valid_pos, return_distance=False)
            for i, v in enumerate(valid_idx):
                global_knn_indices = valid_idx[knn_indices[i]]
                knn[v, :] = np.copy(global_knn_indices)
        except Exception:
            pass

        # Update thge dataframe
        new_row = pd.Series({
            'time': t,
            'inframe': inframe,
            'valid': valid,
            'indices': indices,
            'positions': positions,
            'KNN': knn
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
    def standard_regression(X0, X1):
        return X1 @ np.linalg.pinv(X0)
    if regfun is None:
        regfun = standard_regression

    # Add new columns to the data frame
    df["relF"] = None       # Relative Deformation Gradient (jacobian)
    df["L"] = None          # Velocity Gradient

    # Step through all times in the data set.
    c = 0
    for t in range(n_steps-1):
        if c % 25 == 0:
            tnow = df.loc[t, 'time']
            print(f'Regression: t = {tnow}')
        c += 1
        # Specify the initial and final times
        df0 = df.iloc[t]
        df1 = df.iloc[t+1]
        dt = df1['time'] - df0['time']

        # keep overlapping indices only.
        valid_indices = df0.valid

        # Allocate array of Jacobians for particle i
        Jacobian = np.nan * np.ones((d, d, n_particles))
        L = np.copy(Jacobian)

        # Iterate through the particles in frame at time t and t1
        for i, valid in enumerate(valid_indices):

            if valid:    # Particle persists to time 2.  Do computations
                X0 = []
                X1 = []
                O0 = df0.positions[:][i]        # Origin particle at time 0
                O1 = df1.positions[:][i]        # Origin particle at time 1

                # If all neighbors do not exist at second time
                if all(not valid_indices[j] for j in np.int64(df0.KNN[i][1:])):
                    break

                # Compute X matrices
                for j in np.int64(df0.KNN[i][1:]):
                    if valid_indices[j]:
                        N0 = df0.positions[:][j]    # Neighbor at time 0
                        N1 = df1.positions[:][j]    # Neighbor at time 1
                        X0.append(N0-O0)
                        X1.append(N1-O1)
                X0 = np.array(X0).T     # Format to be [d-by-k]
                X1 = np.array(X1).T

                # regress the function and store the gradient
                newF = regfun(X0, X1)
                if np.isnan(newF[0, 0]):
                    raise ValueError
                newL = (newF - np.eye(d))/dt

                # Store
                Jacobian[:, :, i] = newF
                L[:, :, i] = newL

        # Store
        df.at[t, "relF"] = Jacobian
        df.at[t, "L"]  = L


# Compute the metrics on a given trajectory
def computeMetrics(df, Tmax, metric_list='all'):

    # dimension of the data
    d = np.size(df.positions[0][0])

    # number of particles
    n_particles = np.size(df.indices[0])

    def CalcVorticity(df, idx):

        # initialize
        n_elem = int(d*(d-1)/2)
        vort_list = np.nan * np.ones((n_elem, n_particles))

        # valid elements
        valid = 1-np.array(np.isnan(df['L'][idx]))[0, 0]
        valid_indices = df.indices[0][valid == 1]

        for j in valid_indices:    # Loop the particles that persist

            velGrad = df.loc[idx, 'L'][:, :, j].squeeze()
            vort = np.zeros((n_elem, 1))
            c = 0
            for ii in range(d-1, -1, -1):
                for jj in range(d-1, -1, -1):
                    if jj < ii:
                        vort[c] = velGrad[ii, jj] - velGrad[jj, ii]
                        c += 1

            vort_list[:, j] = vort

        df.at[idx, "vorticity"] = vort_list.squeeze()

    def CalcC(df, idx0, idx1):

        try:
            # Objective calculation of C from compositions
            C_list = np.nan * np.ones((d, d, n_particles))

            # Compute valid particle indices
            valid = 1-np.array(np.isnan(df['relF'][idx0] *
                                        df['relF'][idx1-1]))[0, 0]

            valid_indices = df.indices[0][valid == 1]

            for j in valid_indices:    # Loop the particles that persist

                A = np.eye(d)
                for t in range(idx0, idx1):  # Loop the time indices
                    A = df.loc[t, 'relF'][:, :, j] @ A

                C = A.T @ A
                C_list[:, :, j] = C

            df.at[idx0, "C"]  = C_list
        except Exception:
            return

    def FTLE(df, idx0, idx1, T):

        try:
            ftle_list = np.nan * np.ones((n_particles, 1))

            # Compute valid indices
            valid = 1-np.array(np.isnan(df['relF'][idx0] *
                                        df['relF'][idx1-1]))[0, 0]
            valid_indices = df.indices[0][valid == 1]

            for i in valid_indices:
                C = df.loc[idx0, 'C'][:, :, i]
                lam, _ = la.eig(C)
                ftle = 1./np.abs(T)*np.log(np.sqrt(np.max(lam)))
                ftle_list[i] = ftle
            df.at[idx0, "ftle"]  = ftle_list
        except Exception:
            return

    def LAVD(df, idx0, idx1):

        try:
            lavd_list = np.nan * np.ones((n_particles, 1))

            # Compute valid indices (assumes continuous if true)
            valid = 1-np.array(np.isnan(df['relF'][idx0] *
                                        df['relF'][idx1-1]))[0, 0]

            valid_indices = df.indices[0][valid == 1]
            for i in range(idx0, idx1):
                avg_vort = np.nanmean(df.loc[i, 'vorticity'])
                dt = df.loc[i+1, 'time'] - df.loc[i, 'time']
                for j in valid_indices:
                    lavd_list[j] = np.nansum((lavd_list[j].squeeze(),
                                              np.abs(df.loc[i, 'vorticity'][j]
                                                     - avg_vort)*dt))

            df.at[idx0, "lavd"]  = lavd_list
        except Exception:
            return

    def DRA(df, idx0, idx1):

        # only valid for 2d flows
        assert d == 2, "DRA cannot be computed for 2d flows."

        try:
            dra_list = np.nan * np.ones((n_particles, 1))

            # Compute valid indices (assumes continuous if true)
            valid = 1-np.array(np.isnan(df['relF'][idx0] *
                                        df['relF'][idx1-1]))[0, 0]

            valid_indices = df.indices[0][valid == 1]
            for i in range(idx0, idx1):
                dt = df.loc[i+1, 'time'] - df.loc[i, 'time']
                for j in valid_indices:
                    dra_list[j] = np.nansum((dra_list[j].squeeze(),
                                             df.loc[i, 'vorticity'][j] * dt))

            df.at[idx0, "dra"]  = dra_list
        except Exception:
            return

    # Number of total time intervals
    n_steps = len(df.index)

    # Specify the metrics being computed
    if metric_list == 'all':
        metric_list = ['ftle', 'lavd', 'dra', 'vort']

    # Set columns in the dataframe
    df['vorticity'] = None
    if 'ftle' in metric_list:
        df['C'] = None
        df['ftle'] = None
    if 'lavd' in metric_list:
        df['lavd'] = None
    if 'dra' in metric_list:
        df['dra'] = None

    # At every time step, iterate through the particles and compute
    # the metric fields.

    # Vorticity first:
    for i in range(n_steps-1):
        # Calculate vorticity
        CalcVorticity(df, i)

    # Then finite-time emasures
    c = 0
    for i in range(n_steps-1):

        if c % 25 == 0:
            tnow = df.loc[i, 'time']
            print(f'Metrics: t = {tnow}')
        c += 1

        # For the Lagrangian metrics (over domain Tmax)
        T = 0   # time of Lagrangian computations
        j = 0   # second index
        while T <= Tmax + 0.001:
            if i + j + 1 > n_steps-1:
                T = Tmax + 100      # break the loop
            else:
                dt = df['time'][i+j+1] - df['time'][i+j]
                j += 1
                T += dt

        # Calculate finite-time metrics
        if T >= Tmax:
            if 'ftle' in metric_list:
                CalcC(df, i, i+j)
                FTLE(df, i, i+j, T)
            if 'lavd' in metric_list:
                LAVD(df, i, i+j)
            if 'dra' in metric_list:
                DRA(df, i, i+j)


# standard 2d scattered interpolation
def interpolate_2D(scatteredData, gridvectors, mask=None, method='cubic'):
    # scatteredData should be a list with a tuple containing (x, y) and
    # the values z

    # First, make the target grid to interpolate to.
    xvec = gridvectors[0]
    yvec = gridvectors[1]
    X, Y = np.meshgrid(xvec, yvec)

    # Interpolate the data:
    Z = griddata(scatteredData[0], scatteredData[1], (X, Y), method=method)

    # masking
    if mask is not None:
        Z[mask] = np.nan

    return Z


# Radial basis function interpolation
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
    if mask is not None:
        Z[mask] = np.nan

    return Z


# Compute scalar fields and store to df
def generateFields(df, gridvectors, metric_list='all', approach='interp',
                   method='cubic', mask=None, **kwargs):
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
    c = 0
    for t in range(n_steps-1):  # iterate through times

        if c % 25 == 0:
            tnow = df.loc[t, 'time']
            print(f'Interpolation: t = {tnow}')
        c += 1

        dft = df.loc[t]

        # Format position data to a tuple
        pos = dft['positions']
        x = pos[:, 0]
        y = pos[:, 1]
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

            if data is not None:
                try:
                    if all(item == 0.0 for item in data) or \
                            all(item == np.nan for item in data):
                        scalarFields[s] = None
                    else:
                        scatteredData = get_valid_indices(positions, data)
                        if approach == 'interp':
                            field = interpolate_2D(scatteredData, gridvectors,
                                                   mask=mask, method=method)
                        elif approach == 'rbf':
                            field = rbf_interp(scatteredData, gridvectors,
                                               mask=None, method=method)
                        scalarFields[s] = field
                except Exception:
                    continue

        # Data managment
        df.at[t, 'ScalarFields'] = scalarFields
        fieldsList.append(scalarFields)

    # output value if specified
    if 'outputfields' in kwargs:
        if kwargs['outputfields']:
            return scalarFields
