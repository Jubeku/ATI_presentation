# Plotting module

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import colors
from matplotlib.dates import DateFormatter, MinuteLocator, SecondLocator
from matplotlib.patches import ConnectionPatch
from modules.colormap2d import imshow2d
import os


class MidPointLogNorm(colors.LogNorm):
    
    """
    This class allows to generate a logarithmic colorbar with defined midpoint.
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        colors.LogNorm.__init__(self,vmin=vmin, vmax=vmax, clip=clip)
        self.midpoint=midpoint
    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        x, y = [np.log(self.vmin), np.log(self.midpoint), np.log(self.vmax)], [0, 0.5, 1]
        return np.ma.array(np.interp(np.log(value), x, y), mask=result.mask, copy=False)


def plot_traces(sts,labels,title):

    """
    This function plot streams of seismic signals.
    Several streams can be passed to the function for comparison.

    Input parameters
    sts:     List of N streams (Obspy Stream)
    labels:  List of N labels
    title:   Title of figure
    
    """

    rc('font', size=12.0)
    
    fig, ax = plt.subplots(sts[0].count(), 1, sharex=True, sharey=True,
            figsize=(10,12))
    ax[0].set_title(title)
   
    # Plot each trace of each stream
    for st, label in zip(sts, labels):
        for itr, tr in enumerate(st):
            ilabel = tr.stats.station+'.'+tr.stats.channel[2]+' - '+label
            ax[itr].grid('on',which='both')
            ax[itr].plot(tr.times('matplotlib'), tr.data, lw=0.8, label=ilabel)
            ax[itr].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            if itr is not 0:
                ax[itr].yaxis.get_offset_text().set_visible(False)
            ax[itr].set(ylabel='$v_Z$ (m/s)')
            ax[itr].legend(loc=1)

    ax[itr].set(xlabel='Time (s)')
    # Bring subplots close to each other.
    fig.subplots_adjust(hspace=0.1)
    # Hide x labels and tick labels for all but bottom plot.
    for axi in ax:
        axi.label_outer()
    # Format time axis
    if st[0].stats.npts*st[0].stats.delta > 80 :
        plt.gca().xaxis.set_minor_locator( 
                SecondLocator(bysecond=range(10,60,10)) )
    else:
        plt.gca().xaxis.set_minor_locator( 
                SecondLocator(bysecond=range( 5,60, 5)) )
    plt.gca().xaxis.set_minor_formatter( DateFormatter("%S''") )
    plt.gca().xaxis.set_major_locator( MinuteLocator(byminute=range( 0,60, 1)) )
    plt.gca().xaxis.set_major_formatter( DateFormatter('%H:%M:%S') )
    
    plt.show()


def plot_trace(tr, event):

    """
    This function plots a single seismic trace.

    Input parameters
    tr:      Seismic signal (Obspy Trace)
    
    """

    rc('font', size=12.0)
    
    fig = plt.figure(figsize=(10,4))
    ax  = fig.add_subplot(111)
    ax.set_title('Seismic signal and picked time window (red dashed lines)')
   
    # Plot each trace of each stream
    label = tr.stats.station+'.'+tr.stats.channel[2]
    ax.grid(which='both')
    ax.plot(tr.times('matplotlib'), tr.data, lw=0.8, label=label)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if os.path.isfile('pickedTW_'+event+'.txt'):
        pickTW = np.loadtxt('pickedTW_'+event+'.txt')
        ax.axvline(pickTW[0],color='C3',linestyle='--')
        ax.axvline(pickTW[1],color='C3',linestyle='--')
    ax.legend(loc=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('$v_Z$ (m/s)')
    # Format time axis
    if tr.stats.npts*tr.stats.delta > 80 :
        plt.gca().xaxis.set_minor_locator( 
                SecondLocator(bysecond=range(10,60,10)) )
    else:
        plt.gca().xaxis.set_minor_locator( 
                SecondLocator(bysecond=range( 5,60, 5)) )
    plt.gca().xaxis.set_minor_formatter( DateFormatter("%S''") )
    plt.gca().xaxis.set_major_locator( MinuteLocator(byminute=range( 0,60, 1)) )
    plt.gca().xaxis.set_major_formatter( DateFormatter('%H:%M:%S') )
    
    plt.show()


def plot_LOCA(loc_prob_time, t_rel, swin):

    """
    This functions plots a map of rockfall location probabilities and the 
    corresponding times using a 2D colomap specified in colormap2d.py. 

    Input parameters
    loc_prob_time:  Numpy array with shape (2, nx, ny). The first matrix of 
                    size (nx, ny) corresponds to the times which are represented
                    by color and the second matrix of size (nx, ny) corresponds
                    to the probability represented by lightness.
    t_rel:          Relative time between tstart and tend (picked time window)
    swin:           Step size of sliding time window

    """
    
    rc('font', size=12.0)

    # Load DEM
    topo = np.loadtxt('data/DEM_PF_10m_cut.dat',skiprows=6)
    dx = 10
    dy = 10
    xx = np.arange(0.,2100.+dx,dx)
    yy = np.arange(0.,1800.+dy,dy)
    levels = np.arange(2190., 2650, 20)
    # Load station positions
    stas = np.loadtxt('data/stations.txt')
    stas_name = ['BON', 'BOR', 'DSO', 'SNE']
    xmin =  640.
    xmax = 1840.
    ymin =  400.
    ymax = 1400.

    fig, ax_im = plt.subplots( 1, 1, figsize=(8.,8.))
    # Plot map of probabilities with 2D colormap
    im = imshow2d(loc_prob_time, ax=ax_im, cmap2d='brightwheel',
            huevmin=0., huevmax=t_rel+2*swin,
            lightvmin=0.25, lightvmax=0.55,
            extent=[xmin, xmax, ymin, ymax],
            )
    # Plot contour lines
    co = ax_im.contour(xx, yy, np.flipud(topo),
            levels, colors='C5', alpha=0.5, linewidths=0.5)

    # Plot stations locations and annotate
    bbox = dict(boxstyle='round', fc='1',alpha=0.7,lw=0)
    ax_im.scatter(stas[:,0], stas[:,1],
            marker='^', c='C2', s=300, edgecolor='k',linewidth=0.3)
    for iista, sta in enumerate(stas_name):
        if iista == 0:
            xanno =  -60.
            yanno =   50.
        elif iista == 1:
            xanno =  -60.
            yanno =  -100.
        elif iista == 2:
            xanno =  -60.
            yanno =  -100.
        elif iista == 3:
            xanno = -200.
            yanno =  -20.
        anno = (stas_name[iista])
        ax_im.annotate(anno, xy=(stas[iista,0],stas[iista,1]),
                xytext=(stas[iista,0]+xanno,stas[iista,1]+yanno),
                size=24, color='C2')
    
    # Set limits
    ax_im.set_xlim(xx[0]+400., xx[-1]-200.)
    ax_im.set_ylim(yy[0]+300., yy[-1]-250.)
    ax_im.set_xticks([]) 
    ax_im.set_yticks([]) 
   
    plt.show()
   

def plot_Tr_Cb(tr, tstart, tend, t_rel, swin):
    
    """
    This function plots a colorbar for the figure plotted with  plot_LOCA().
    A seismic trace is added to visualize the time window used for the 
    localization.

    Input parameters
    tr:     Seismic trace (Obspy Trace)
    tstart: Start of picked time window
    tend:   End of picked time window
    t_rel:  Relative time between tstart and tend (picked time window)
    swin:   Step size of sliding time window

    """
    rc('font', size=18.0)
    
    # Prepare colorbar
    times = np.arange(0., t_rel+swin, swin)
    dp = 0.01
    prob  = np.arange(0., 1.+dp    , 0.01 )
    mesht = np.tile(times, (len(prob),1))
    meshp = np.tile(prob, (len(times),1))
    color = np.array([mesht, meshp.transpose()])

    
    fig   = plt.figure(figsize=(7, 2))
    ax    = fig.add_axes([0.1,0.35,0.8,0.6])
    ax_cb = fig.add_axes([0.1,0.1,0.8,0.15])
    # Plot seismic signal and picked time window
    ax.grid(which='both')
    ax.plot(tr.times('matplotlib'), tr.normalize().data, lw=0.8, label=tr.stats.station)
    ax.axvline(tstart,color='k',linestyle='--')
    ax.axvline(tend,  color='k',linestyle='--')
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.legend(loc=1)
    ax.axis('off')
    # Plot colorbar
    im = imshow2d(color, ax=ax_cb, cmap2d='brightwheel',
                    huevmin=0., huevmax=t_rel+2*swin,
                    lightvmin=0., lightvmax=1.,
                    origin = 'lower',
                    extent=[0., t_rel, 0., 1.],
                    aspect='auto'
                    )
    ax_cb.set_xlabel('Time (s)') 
    ax_cb.set_ylabel('PDF') 
    
    # Draw lines linking the 2 plots
    ymin, ymax = ax.get_ylim()
    xyA = (tstart, ymin)
    xyB = (0., 1.)
    con1 = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA="data", coordsB="data",
                      axesA=ax, axesB=ax_cb, color="k", ls='--', lw=1.5)
    ax.add_artist(con1)
    xyA = (tend, ymin)
    xyB = (t_rel, 1.)
    con2 = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA="data", coordsB="data",
                      axesA=ax, axesB=ax_cb, color="k", ls='--', lw=1.5)
    ax.add_artist(con2)

    plt.show()


def plot_ratios_on_map(data, ref_sta, sec_sta):

    """
    This functions plots a map energy ratios between two stations. 

    Input parameters
    data:    Pandas dataframe with energy values
    ref_sta: Name of dataframe column corresponding to the reference station
    sec_sta: Name of dataframe column corresponding to the second station
    ax_im  : Matplotlib figure axes instance
    
    """
    
    rc('font', size=12.0)

    # Load DEM
    topo = np.loadtxt('data/DEM_PF_10m_cut.dat',skiprows=6)
    dx = 10
    dy = 10
    xx = np.arange(0.,2100.+dx,dx)
    yy = np.arange(0.,1800.+dy,dy)
    levels = np.arange(2190., 2650, 20)
    # Load station positions
    stas = np.loadtxt('data/stations.txt')
    stas_name = ['BON', 'BOR', 'DSO', 'SNE']
    xmin =  640.
    xmax = 1840.
    ymin =  400.
    ymax = 1400.
    
    # Calculate ratios and reshape
    ratios = np.array(data[sec_sta]/data[ref_sta])
    xdim = 121
    ydim = 101
    ratios = np.reshape(ratios,(ydim,xdim))
    
    fig    = plt.figure(figsize=(10,5))
    ax_im  = fig.add_subplot(111)
    # Plot map of probabilities with 2D colormap
    im = ax_im.imshow(np.flipud(ratios),
            norm=MidPointLogNorm(vmin=0.05,vmax=20.,midpoint=1),
            interpolation='bilinear', cmap='bwr',
            extent=[xmin,xmax,ymin,ymax])
    clabel = '$E_{topo}/E_{flat}$'

    # Plot contour lines
    co = ax_im.contour(xx, yy, np.flipud(topo),
            levels, colors='C5', alpha=0.5, linewidths=0.5)

    # Plot stations locations and annotate
    bbox = dict(boxstyle='round', fc='1',alpha=0.7,lw=0)
    for iista, sta in enumerate(stas_name):
        if iista == 0:
            xanno =  -90.
            yanno =   50.
        elif iista == 1:
            xanno =  -90.
            yanno =  -150.
        elif iista == 2:
            xanno =   50.
            yanno =  -80.
        elif iista == 3:
            xanno = -250.
            yanno =  -20.
        anno = (stas_name[iista])
        if sta == ref_sta[-3:] or sta == sec_sta[-3:]:
            color='C2'
        else:
            color='C7'
        ax_im.scatter(stas[iista,0], stas[iista,1],
                marker='^', c=color, s=300, edgecolor='k',linewidth=0.3)
        ax_im.annotate(anno, xy=(stas[iista,0],stas[iista,1]),
                xytext=(stas[iista,0]+xanno,stas[iista,1]+yanno),
                size=24, color=color)

    
    # Set limits
    ax_im.set_xlim(xx[0]+400., xx[-1]-200.)
    ax_im.set_ylim(yy[0]+300., yy[-1]-250.)
    ax_im.set(xlabel='x', ylabel='y')
   
    # Create colorbar
    ycol = np.arange(0.05, 0.1, 0.01)
    ycol = np.append(ycol, np.arange(0.1, 1., 0.1))
    ycol = np.append(ycol, np.arange(1., 10., 1.))
    ycol = np.append(ycol, np.arange(10., 25., 10.))
    minorticks = im.norm(ycol)
    cb    = fig.colorbar(im)
    cb.ax.yaxis.set_ticks(minorticks, minor=True)
    cb.ax.set_ylabel('$E_{topo}/E_{flat}$')
    
    # Set title
    ax_im.set_title(sec_sta+'/'+ref_sta)

    plt.show()