
# General
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
mpl.rcParams['figure.figsize'] = [15, 15]
mpl.rcParams['font.sans-serif'] = "Times New Roman"
mpl.rcParams['font.family'] = "serif"

# Custom Colors
LIGHTGRAY = [0.85, 0.85, 0.85]

# Custom Colormaps
highContrast_b2r_black = LinearSegmentedColormap.from_list(
    'highContrast_b2r_black', [
        (0.0, 'darkblue'), (0.1, 'blue'), (0.4, 'cyan'),
        (0.5, 'black'), (0.6, 'yellow'),
        (0.9, 'red'), (1.0, 'darkred')])
mpl.colormaps.register(highContrast_b2r_black)

b2r_black = LinearSegmentedColormap.from_list('b2r_black', [
    (0.0, 'blue'), (0.5, 'black'), (1.0, 'red')])
mpl.colormaps.register(b2r_black)

highContrast_white = LinearSegmentedColormap.from_list(
    'highContrast_b2r_white', [
        (0.0, 'cyan'), (0.2, 'blue'), (0.3, 'darkblue'),
        (0.5, 'white'), (0.7, 'darkred'),
        (0.8, 'red'), (1.0, 'yellow')])
mpl.colormaps.register(highContrast_white)


# Function to invert a colormap
def invert_cmap(cmap_name):
    cmap = mpl.colormaps[cmap_name]
    rgb = cmap(np.linspace(0, 1, 10))[:, :3]
    rgb_i = 1-rgb
    outmap = LinearSegmentedColormap.from_list(cmap_name, rgb_i, N=256)
    mpl.colormaps.register(outmap)
    return outmap


gray2cool = LinearSegmentedColormap.from_list(
    'gray2cool', [
        (0.0, 'white'), (0.5, 'black'),
        (0.65, 'darkblue'), (0.85, 'blue'), (1, 'cyan')])
mpl.colormaps.register(gray2cool)
mpl.colormaps.register(cmap=gray2cool.reversed())

gray2blue = LinearSegmentedColormap.from_list(
    'gray2blue', [
        (0.0, 'white'), (0.25, 'gray'), (0.7, 'blue'),
        (0.9, 'darkblue'), (1, 'midnightblue')])
mpl.colormaps.register(gray2blue)
mpl.colormaps.register(cmap=gray2blue.reversed())

gray2hot = LinearSegmentedColormap.from_list(
    'gray2hot', [
        (0.0, 'white'), (0.5, 'black'),
        (0.65, 'darkred'), (0.85, 'red'), (1, 'yellow')])
mpl.colormaps.register(gray2hot)
mpl.colormaps.register(cmap=gray2hot.reversed())

gray2red = LinearSegmentedColormap.from_list(
    'gray2red', [
        (0.0, 'white'), (0.25, 'gray'), (0.7, 'red'),
        (0.9, 'firebrick'), (1, 'maroon')])
mpl.colormaps.register(gray2red)
mpl.colormaps.register(cmap=gray2red.reversed())

cool2gray2hot = LinearSegmentedColormap.from_list(
    'cool2gray2hot', [
        (0.0, 'cyan'), (0.05, 'blue'), (0.1, 'darkblue'),
        (0.2, 'black'), (0.5, 'white'), (0.8, 'black'),
        (0.9, 'darkred'), (0.95, 'red'), (1, 'yellow')])
mpl.colormaps.register(cool2gray2hot)
mpl.colormaps.register(cmap=cool2gray2hot.reversed())


def stitchColormaps(data, cm1, cm2, stitch_at='mid', name='stitched_cmap'):

    # Get the data midpoint
    dmin = np.nanmin(data)
    dmax = np.nanmax(data)
    if stitch_at == 'mid':
        dmid = (dmin+dmax)/2
        prop = 0.5
    else:
        if stitch_at > dmax or stitch_at < dmin:
            dmid = (dmin+dmax)/2
            prop = 0.5
        else:
            dmid = stitch_at
            prop = np.abs(dmid-dmin)/np.abs(dmax-dmin)

    # Vectors based on proportion
    vec1 = np.linspace(0, 1, int(np.round(512*prop)))
    vec2 = np.linspace(0, 1, int(np.round(512*(1-prop))))

    # Get the colors of the colormaps
    colors1 = plt.cm.get_cmap(cm1)
    colors2 = plt.cm.get_cmap(cm2)
    cmap1 = colors1(vec1)
    cmap2 = colors2(vec2)

    # stack them and create a colormap
    cmap_vec = np.vstack((cmap1, cmap2))
    mymap = LinearSegmentedColormap.from_list(name, cmap_vec)

    return mymap


# A function for plotting all metrics from random data.
def plotAllMetrics(df, xvec, yvec, tstep=100, particles=True):
    X, Y = np.meshgrid(xvec, yvec)
    xlim = [np.min(xvec), np.max(xvec)]
    ylim = [np.min(yvec), np.max(yvec)]

    tstep = tstep

    x = df.loc[tstep, 'positions'][: 0]
    y = df.loc[tstep, 'positions'][: 1]

    ftle = np.squeeze(df.loc[tstep, 'ScalarFields']['ftle'])
    lavd = np.squeeze(df.loc[tstep, 'ScalarFields']['lavd'])
    dra = np.squeeze(df.loc[tstep, 'ScalarFields']['dra'])
    vort = np.squeeze(df.loc[tstep, 'ScalarFields']['vort'])

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=[12, 6])
    plt.subplots_adjust(hspace=0.05, wspace=0.15)

    clim = [0, np.nanmax(ftle)]
    ftleim = axs[0, 0].pcolormesh(X, Y, ftle, cmap='gray2hot',
                                  vmin=clim[0], vmax=clim[1])
    if particles:
        axs[0, 0].scatter(x, y, s=2, c=[[0.6, 0.6, 0.6, 0.7]])
    axs[0, 0].axis('scaled')
    axs[0, 0].set_xlim(xlim)
    axs[0, 0].set_ylim(ylim)
    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(ftleim, cax=cax)
    axs[0, 0].set_title('FTLE')

    clim = [0, np.nanmax(lavd)]
    lavdim = axs[0, 1].pcolormesh(X, Y, lavd, cmap='gray2cool',
                                  vmin=clim[0], vmax=clim[1])
    if particles:
        axs[0, 1].scatter(x, y, s=2, c=[[0.6, 0.6, 0.6, 0.7]])
    axs[0, 1].axis('scaled')
    axs[0, 1].set_xlim(xlim)
    axs[0, 1].set_ylim(ylim)
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(lavdim, cax=cax)
    axs[0, 1].set_title('LAVD')

    clim = [-np.nanmax(np.abs(dra)), np.nanmax(np.abs(dra))]
    torim = axs[1, 0].pcolormesh(X, Y, dra, cmap='highContrast_b2r_white',
                                 vmin=clim[0], vmax=clim[1])
    if particles:
        axs[1, 0].scatter(x, y, s=2, c=[[0.6, 0.6, 0.6, 0.7]])
    axs[1, 0].axis('scaled')
    axs[1, 0].set_xlim(xlim)
    axs[1, 0].set_ylim(ylim)
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(torim, cax=cax)
    axs[1, 0].set_title('DRA')

    clim = [-np.nanmax(np.abs(vort)), np.nanmax(np.abs(vort))]
    vortim = axs[1, 1].pcolormesh(X, Y, vort, cmap='bwr',
                                  vmin=clim[0], vmax=clim[1])
    if particles:
        axs[1, 1].scatter(x, y, s=2, c=[[0.6, 0.6, 0.6, 0.7]])
    axs[1, 1].axis('scaled')
    axs[1, 1].set_xlim(xlim)
    axs[1, 1].set_ylim(ylim)
    divider = make_axes_locatable(axs[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(vortim, cax=cax)
    axs[1, 1].set_title('Vorticity')

    fig.show()


# Plot trajectories of particles
def plot_trajectories(trajectories, cmap_name='rainbow'):

    n_particles, n_times, dim = np.shape(trajectories)

    # Colors for plotting
    cmap = plt.cm.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, n_particles))

    # Create a new figure
    fig, ax = plt.subplots(1, 1)

    # Set the x and y limits to be the same
    ax.axis('scaled')
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 1])

    for i in range(n_particles):
        x_vals = trajectories[i, :, 0]
        y_vals = trajectories[i, :, 1]
        ax.plot(x_vals, y_vals, color=colors[i, :])

    plt.show()
