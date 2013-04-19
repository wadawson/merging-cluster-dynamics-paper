'''
This program takes the output pickle arrays of TSM.py and creates some plots and
statistics.  It is largely based on plotTSM.py

The main function of this program is to generate report quality plots,
especially a covariance array plot
'''
from __future__ import division
import pylab
import numpy
import pickle
from scipy.stats import norm
from scipy.special import erf
from astrostats import biweightLoc,bcpcl
import cosmo
import tools

#prefix from TSM_MCcalc.py
prefix = '/Users/dawson/Documents/Research/DLSCL09162953/TimeSinceMerger/newTSMcode/run_v11/run_v11_'
index = ('0','1','2','3','4','5','6','7','8','9')
#Histogram bins
N_bins_2d = 100
N_bins_1d = 400
N_bins_TSM = 45
N_bins_alpha = 90


def loadcombo(prefix,index,suffix):
    array = []
    for i in index:
        filename = prefix+i+'_'+suffix+'.pickle'
        #read in the pickled array
        F = open(filename)
        tmp = pickle.load(F)
        F.close()
        array = numpy.append(array,tmp)
    return array

# Create the data arrays
m_1 = loadcombo(prefix,index,'m_1')
m_2 = loadcombo(prefix,index,'m_2')
z_1 = loadcombo(prefix,index,'z_1')
z_2 = loadcombo(prefix,index,'z_2')
d_proj = loadcombo(prefix,index,'d_proj')
v_rad_obs = loadcombo(prefix,index,'v_rad_obs')
alpha = loadcombo(prefix,index,'alpha')
v_3d_obs = loadcombo(prefix,index,'v_3d_obs')
d_3d = loadcombo(prefix,index,'d_3d')
v_3d_col = loadcombo(prefix,index,'v_3d_col')
d_max = loadcombo(prefix,index,'d_max')
TSM_0 = loadcombo(prefix,index,'TSM_0')
TSM_1 = loadcombo(prefix,index,'TSM_1')
T = loadcombo(prefix,index,'T')
prob = loadcombo(prefix,index,'prob')

# for some reason 4 of the T, and thus prob, elements have nan values. Need to
# remove these cases from all arrays
mask_nan = ~numpy.isnan(T)
m_1 = m_1[mask_nan]
m_2 = m_2[mask_nan]
z_1 = z_1[mask_nan]
z_2 = z_2[mask_nan]
d_proj = d_proj[mask_nan]
v_rad_obs = v_rad_obs[mask_nan]
alpha = alpha[mask_nan]
v_3d_obs = v_3d_obs[mask_nan]
d_3d = d_3d[mask_nan]
v_3d_col = v_3d_col[mask_nan]
d_max = d_max[mask_nan]
TSM_0 = TSM_0[mask_nan]
TSM_1 = TSM_1[mask_nan]
T = T[mask_nan]
prob = prob[mask_nan]

# Put masses in units of 10^14 solar masses
m_1 /= 1e14
m_2 /= 1e14

# Create the TSM_1 < age of universe mask
mask_TSM_1 = TSM_1 < cosmo.age(0)

def histplot2d(x,y,prefix,prob=None,N_bins=100,histrange=None,x_lim=None,y_lim=None,x_label=None,y_label=None,legend=None):
    '''
    Input:
    x = [1D array of N floats]
    y = [1D array of N floats]
    prefix = [string] prefix of output file
    prob = [None] or [1D array of N floats] weights to apply to each (x,y) pair
    N_bins = [integer] the number of bins in the x and y directions   
    histrange = [None] or [array of floats: (x_min,x_max,y_min,y_max)] the range
        over which to perform the 2D histogram and estimate the confidence
        intervals
    x_lim = [None] or [array of floats: (x_min,x_max)] min and max of the range
        to plot
    y_lim = [None] or [array of floats: (x_min,x_max)] min and max of the range
        to plot
    x_label = [None] or [string] the plot's x-axis label
    y_label = [None] or [string] the plot's y-axis label
    legend = [None] or [True] whether to display a legend or not
    '''
    #Create the confidence interval plot
    if histrange == None:
        if prob != None:
            H, xedges, yedges = numpy.histogram2d(x,y,bins=N_bins,weights=prob)
        elif prob == None:
            H, xedges, yedges = numpy.histogram2d(x,y,bins=N_bins)
    else:
        if prob != None:
            H, xedges, yedges = numpy.histogram2d(x,y,bins=N_bins,range=[[histrange[0],histrange[1]],[histrange[2],histrange[3]]],weights=prob)
        elif prob == None:
            H, xedges, yedges = numpy.histogram2d(x,y,bins=N_bins,range=[[histrange[0],histrange[1]],[histrange[2],histrange[3]]])
    H = numpy.transpose(H)
    #Flatten H
    h = numpy.reshape(H,(N_bins**2))
    #Sort h from smallest to largest
    index = numpy.argsort(h)
    h = h[index]
    h_sum = numpy.sum(h)
    # Find the 2 and 1 sigma levels of the MC hist
    for j in numpy.arange(numpy.size(h)):
        if j == 0:
            runsum = h[j]
        else:
            runsum += h[j]
        if runsum/h_sum <= 0.05:
            #then store the value of N at the 2sigma level
            h_2sigma = h[j]
        if runsum/h_sum <= 0.32:
            #then store the value of N at the 1sigma level
            h_1sigma = h[j]
    # Create the contour plot using the 2Dhist info
    # define pixel values to be at the center of the bins
    x = xedges[:-1]+(xedges[1]-xedges[0])/2
    y = yedges[:-1]+(yedges[1]-yedges[0])/2
    X, Y = numpy.meshgrid(x,y)
    
    fig = pylab.figure()
    # Countours
    CS = pylab.contour(X,Y,H,(h_2sigma,h_1sigma),linewidths=(2,2)) 
    # imshow
    #im = pylab.imshow(H,cmap=pylab.cm.gray)
    pylab.pcolor(X,Y,H,cmap=pylab.cm.gray_r)
    
    if x_label != None:
        pylab.xlabel(x_label,fontsize=14)
    if y_label != None:    
        pylab.ylabel(y_label,fontsize=14)
    if x_lim != None:
        pylab.xlim(x_lim)
    if y_lim != None:
        pylab.ylim(y_lim)
    if legend != None:
        # Dummy lines for legend
        pylab.plot((0,1),(0,1),c='#800000',linewidth=2,label=('68%'))
        pylab.plot((0,1),(0,1),c='#0000A0',linewidth=2,label=('95%'))      
        pylab.legend(scatterpoints=1)
    fontsize=14
    ax = pylab.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    
    filename = prefix+'_histplot2d'
    pylab.savefig(filename)
    
    return fig

def histplot1d(x,prefix,prob=None,N_bins=100,histrange=None,x_lim=None,y_lim=None,x_label=None,y_label=None,legend=None):
    '''
    '''
    fig = pylab.figure()
    hist, binedges, tmp = pylab.hist(x,bins=N_bins,histtype='step',weights=prob,range=histrange,color='k',linewidth=2)
    # Calculate the location and %confidence intervals
    # Since my location and confidence calculations can't take weighted data I
    # need to use the weighted histogram data in the calculations
    for i in numpy.arange(N_bins):
        if i == 0:
            x_binned = numpy.ones(hist[i])*(binedges[i]+binedges[i+1])/2
        else:
            x_temp = numpy.ones(hist[i])*(binedges[i]+binedges[i+1])/2
            x_binned = numpy.concatenate((x_binned,x_temp))
    loc = biweightLoc(x_binned)
    ll_68, ul_68 = bcpcl(loc,x_binned,1)
    ll_95, ul_95 = bcpcl(loc,x_binned,2)
    # Create location and confidence interval line plots
    pylab.plot((loc,loc),(pylab.ylim()[0],pylab.ylim()[1]),'--k',linewidth=2,label='$C_{BI}$')
    pylab.plot((ll_68,ll_68),(pylab.ylim()[0],pylab.ylim()[1]),'-.',linewidth=2,color='#800000',label='68% $IC_{B_{BI}}$')
    pylab.plot((ul_68,ul_68),(pylab.ylim()[0],pylab.ylim()[1]),'-.',linewidth=2,color='#800000')
    pylab.plot((ll_95,ll_95),(pylab.ylim()[0],pylab.ylim()[1]),':',linewidth=2,color='#0000A0',label='95% $IC_{B_{BI}}$')
    pylab.plot((ul_95,ul_95),(pylab.ylim()[0],pylab.ylim()[1]),':',linewidth=2,color='#0000A0')
    
    if x_label != None:
        pylab.xlabel(x_label,fontsize=14)
    if y_label != None:    
        pylab.ylabel(y_label,fontsize=14)
    if x_lim != None:
        pylab.xlim(x_lim)
    if y_lim != None:
        pylab.ylim(y_lim)
    if legend != None:
        pylab.legend()
    #fontsize=14
    #ax = pylab.gca()
    #for tick in ax.xaxis.get_major_ticks():
        #tick.label1.set_fontsize(fontsize)

    filename = prefix+'_histplot1D'
    pylab.savefig(filename)
    
    print '{0}, {1:0.4f}, {2:0.4f}, {3:0.4f}, {4:0.4f}, {5:0.4f}'.format(prefix,loc,ll_68,ul_68,ll_95,ul_95)
    return fig

def histplot1d_part(ax,x,prob=None,N_bins=100,histrange=None,x_lim=None,y_lim=None):
    '''
    This take the additional value of an array axes. for use with subplots
    '''
    hist, binedges, tmp = ax.hist(x,bins=N_bins,histtype='step',weights=prob,range=histrange,color='k',linewidth=1)
    if y_lim != None:
        ax.set_ylim((y_lim[0],y_lim[1]))    
    # Calculate the location and %confidence intervals
    # Since my location and confidence calculations can't take weighted data I
    # need to use the weighted histogram data in the calculations
    for i in numpy.arange(N_bins):
        if i == 0:
            x_binned = numpy.ones(hist[i])*(binedges[i]+binedges[i+1])/2
        elif numpy.size(x_binned) == 0:
            x_binned = numpy.ones(hist[i])*(binedges[i]+binedges[i+1])/2
        else:
            x_temp = numpy.ones(hist[i])*(binedges[i]+binedges[i+1])/2
            x_binned = numpy.concatenate((x_binned,x_temp))
    loc = biweightLoc(x_binned)
    ll_68, ul_68 = bcpcl(loc,x_binned,1)
    ll_95, ul_95 = bcpcl(loc,x_binned,2)

    mask_68 = numpy.logical_and(binedges>=ll_68,binedges<=ul_68)
    mask_95 = numpy.logical_and(binedges>=ll_95,binedges<=ul_95)

    ax.hist(x,bins=binedges[mask_95],weights=prob,histtype='stepfilled',linewidth=0,color=(158/255.,202/255.,225/255.))
    ax.hist(x,bins=binedges[mask_68],weights=prob,histtype='stepfilled',linewidth=0,color=(49/255.,130/255.,189/255.))
    
    # determine the height of the histogram at the loc
    #determine the binedge to the left and right of the location
    idx = tools.argnearest(binedges,loc)
    near = tools.nearest(binedges,loc)
    if loc >= near:
        y_loc = hist[idx]
    else:
        y_loc = hist[idx-1]
    
    # Create location and confidence interval line plots
    ax.plot((loc,loc),(pylab.ylim()[0],y_loc),'--k',linewidth=2,label='$C_{BI}$')
    
    
    # replot the 1D histogram so that the line is not covered by colored sections
    hist, binedges, tmp = ax.hist(x,bins=N_bins,histtype='step',weights=prob,range=histrange,color='k',linewidth=2)
    
    if x_lim != None:
        ax.set_xlim(x_lim)
    if y_lim != None:
        ax.set_ylim(y_lim)
    return loc, ll_68, ul_68, ll_95, ul_95


def histplot2d_part(ax,x,y,prob=None,N_bins=100,histrange=None,x_lim=None,y_lim=None):
    '''
    This take the additional value of an array axes. for use with subplots
    Input:
    x = [1D array of N floats]
    y = [1D array of N floats]
    prefix = [string] prefix of output file
    prob = [None] or [1D array of N floats] weights to apply to each (x,y) pair
    N_bins = [integer] the number of bins in the x and y directions   
    histrange = [None] or [array of floats: (x_min,x_max,y_min,y_max)] the range
        over which to perform the 2D histogram and estimate the confidence
        intervals
    x_lim = [None] or [array of floats: (x_min,x_max)] min and max of the range
        to plot
    y_lim = [None] or [array of floats: (x_min,x_max)] min and max of the range
        to plot
    x_label = [None] or [string] the plot's x-axis label
    y_label = [None] or [string] the plot's y-axis label
    legend = [None] or [True] whether to display a legend or not
    '''
    #Create the confidence interval plot
    if histrange == None:
        if prob != None:
            H, xedges, yedges = numpy.histogram2d(x,y,bins=N_bins,weights=prob)
        elif prob == None:
            H, xedges, yedges = numpy.histogram2d(x,y,bins=N_bins)
    else:
        if prob != None:
            H, xedges, yedges = numpy.histogram2d(x,y,bins=N_bins,range=[[histrange[0],histrange[1]],[histrange[2],histrange[3]]],weights=prob)
        elif prob == None:
            H, xedges, yedges = numpy.histogram2d(x,y,bins=N_bins,range=[[histrange[0],histrange[1]],[histrange[2],histrange[3]]])
    H = numpy.transpose(H)
    #Flatten H
    h = numpy.reshape(H,(N_bins**2))
    #Sort h from smallest to largest
    index = numpy.argsort(h)
    h = h[index]
    h_sum = numpy.sum(h)
    # Find the 2 and 1 sigma levels of the MC hist
    for j in numpy.arange(numpy.size(h)):
        if j == 0:
            runsum = h[j]
        else:
            runsum += h[j]
        if runsum/h_sum <= 0.05:
            #then store the value of N at the 2sigma level
            h_2sigma = h[j]
        if runsum/h_sum <= 0.32:
            #then store the value of N at the 1sigma level
            h_1sigma = h[j]
    # Create the contour plot using the 2Dhist info
    # define pixel values to be at the center of the bins
    x = xedges[:-1]+(xedges[1]-xedges[0])/2
    y = yedges[:-1]+(yedges[1]-yedges[0])/2
    X, Y = numpy.meshgrid(x,y)
    
    # Countours
    CS = ax.contour(X,Y,H,(h_2sigma,h_1sigma),linewidths=(2,2),colors=((158/255.,202/255.,225/255.),(49/255.,130/255.,189/255.))) 
    # imshow
    #im = ax.imshow(H,cmap=ax.cm.gray)
    ax.pcolor(X,Y,H,cmap=pylab.cm.gray_r)
    
    if x_lim != None:
        ax.set_xlim(x_lim)
    if y_lim != None:
        ax.set_ylim(y_lim)

def histplot2dTSC(x,y,prefix,prob=None,N_bins=100,histrange=None,x_lim=None,y_lim=None,x_label=None,y_label=None,legend=None):
    '''
    Input:
    x = [1D array of N floats]
    y = [1D array of N floats]
    prefix = [string] prefix of output file
    prob = [None] or [1D array of N floats] weights to apply to each (x,y) pair
    N_bins = [integer] the number of bins in the x and y directions   
    histrange = [None] or [array of floats: (x_min,x_max,y_min,y_max)] the range
        over which to perform the 2D histogram and estimate the confidence
        intervals
    x_lim = [None] or [array of floats: (x_min,x_max)] min and max of the range
        to plot
    y_lim = [None] or [array of floats: (x_min,x_max)] min and max of the range
        to plot
    x_label = [None] or [string] the plot's x-axis label
    y_label = [None] or [string] the plot's y-axis label
    legend = [None] or [True] whether to display a legend or not
    '''
    # Input calculated v and t parameters for other Dissociative Mergers
    v_bullet_analytic = 3400
    t_bullet_analytic = 0.218
    
    v_bullet_sf07 = 3400
    t_bullet_sf07 = 0.18
    
    v_macs = 2000
    t_macs = 0.255
    
    v_a520 = 2300
    t_a520 = 0.24
    
    v_pandora = 4045
    t_pandora = 0.162   
    
    #Create the confidence interval plot
    if histrange == None:
        if prob != None:
            H, xedges, yedges = numpy.histogram2d(x,y,bins=N_bins,weights=prob)
        elif prob == None:
            H, xedges, yedges = numpy.histogram2d(x,y,bins=N_bins)
    else:
        if prob != None:
            H, xedges, yedges = numpy.histogram2d(x,y,bins=N_bins,range=[[histrange[0],histrange[1]],[histrange[2],histrange[3]]],weights=prob)
        elif prob == None:
            H, xedges, yedges = numpy.histogram2d(x,y,bins=N_bins,range=[[histrange[0],histrange[1]],[histrange[2],histrange[3]]])
    H = numpy.transpose(H)
    #Flatten H
    h = numpy.reshape(H,(N_bins**2))
    #Sort h from smallest to largest
    index = numpy.argsort(h)
    h = h[index]
    h_sum = numpy.sum(h)
    # Find the 2 and 1 sigma levels of the MC hist
    for j in numpy.arange(numpy.size(h)):
        if j == 0:
            runsum = h[j]
        else:
            runsum += h[j]
        if runsum/h_sum <= 0.05:
            #then store the value of N at the 2sigma level
            h_2sigma = h[j]
        if runsum/h_sum <= 0.32:
            #then store the value of N at the 1sigma level
            h_1sigma = h[j]
    # Create the contour plot using the 2Dhist info
    # define pixel values to be at the center of the bins
    x = xedges[:-1]+(xedges[1]-xedges[0])/2
    y = yedges[:-1]+(yedges[1]-yedges[0])/2
    X, Y = numpy.meshgrid(x,y)
    
    fig = pylab.figure()
    # Countours
    CS = pylab.contour(X,Y,H,(h_2sigma,h_1sigma),linewidths=(2,2)) 
    # imshow
    #im = pylab.imshow(H,cmap=pylab.cm.gray)
    pylab.pcolor(X,Y,H,cmap=pylab.cm.gray_r)

    # Data points for other dissociative mergers
    pylab.scatter(v_bullet_sf07,t_bullet_sf07,s=140,c='k',marker='d',label="Bullet SF07")
    #pylab.scatter(v_macs,t_macs,s=140, c='0.25',edgecolor='0.25', marker='^',label='MACS J0025.4')
    #pylab.scatter(v_a520,t_a520,s=140,c='0.25',edgecolor='0.25',marker='o',label='A520')
    #pylab.scatter(v_pandora,t_pandora,s=140,c='0.25',edgecolor='0.25',marker='p',label='A2744')
    
    if x_label != None:
        pylab.xlabel(x_label,fontsize=14)
    if y_label != None:    
        pylab.ylabel(y_label,fontsize=14)
    if x_lim != None:
        pylab.xlim(x_lim)
    if y_lim != None:
        pylab.ylim(y_lim)
    if legend != None:
        # Dummy lines for legend
        pylab.plot((0,1),(0,1),c='#800000',linewidth=2,label=('68%'))
        pylab.plot((0,1),(0,1),c='#0000A0',linewidth=2,label=('95%'))      
        pylab.legend(scatterpoints=1)
    fontsize=14
    ax = pylab.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    
    filename = prefix+'_histplot2dTSC'
    pylab.savefig(filename)

m1_lim = (0,7)
m2_lim = (0,7)
z1_lim = (0.528,0.536)
z2_lim = (0.528,0.536)
dproj_lim = (0.5,1.5)
alpha_lim = (0,90)
d3d_lim = (0,6)
dmax_lim =(0,8)
v3dcol_lim = (1500,3000)
v3dobs_lim = (0,2000)
tsc0_lim = (0,6)
tsc1_lim = (0,14)
t_lim = (0,25)
histrange_dmax=(0,10)
histrange_T=(0,30)

ylab = None
# Use probability
probinput = prob
#probinput = None

## Create the input-input figure array
#print 'creating input-input figure'
#f, axarr = pylab.subplots(5, 5)
#f.subplots_adjust(wspace=0, hspace=0)
## Remove the [0,0] plot's y tickmarks
#pylab.setp([axarr[0,0].get_yticklabels()], visible=False)
#for i in numpy.arange(4):
    ## Remove unnecessary subplots
    #pylab.setp([a.get_axes() for a in axarr[i,i+1:]], visible=False)
    ## Remove the unecessary row axes labels
    #pylab.setp([a.get_xticklabels() for a in axarr[i,:]], visible=False)
    ## Remove the unecessary column axes labels
    #pylab.setp([a.get_yticklabels() for a in axarr[i+1,1:]], visible=False)

## rotate the redshift x-labels by 30 degrees so they don't bunch
#labels = axarr[4,2].get_xticklabels() 
#for label in labels: 
    #label.set_rotation(-90) 
#labels = axarr[4,3].get_xticklabels() 
#for label in labels: 
    #label.set_rotation(-90)

## Create the y-axes labels
#axarr[1,0].set_ylabel('$M_{2} (10^{14} M_\odot)$',size=16)
#axarr[2,0].set_ylabel('$z_1$',size=16)
#axarr[3,0].set_ylabel('$z_2$',size=16)
#axarr[4,0].set_ylabel('$d_{proj}(t_{obs})$',size=16)
## Create the x-axes labels
#axarr[4,0].set_xlabel('$M_{1} (10^{14} M_\odot)$',size=16)
#axarr[4,1].set_xlabel('$M_{2} (10^{14} M_\odot)$',size=16)
#axarr[4,2].set_xlabel('$z_1$',size=16)
#axarr[4,3].set_xlabel('$z_2$',size=16)
#axarr[4,4].set_xlabel('$d_{proj}(t_{obs})$ (Mpc)',size=16)

## First Column
#histplot1d_part(axarr[0,0],m_1,prob=probinput,N_bins=N_bins_1d,histrange=None,x_lim=m1_lim,y_lim=None)

#histplot2d_part(axarr[1,0],m_1,m_2,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=m1_lim,y_lim=m2_lim)

#histplot2d_part(axarr[2,0],m_1,z_1,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=m1_lim,y_lim=z1_lim)

#histplot2d_part(axarr[3,0],m_1,z_2,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=m1_lim,y_lim=z2_lim)

#histplot2d_part(axarr[4,0],m_1,d_proj,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=m1_lim,y_lim=dproj_lim)

## Second Column
#histplot1d_part(axarr[1,1],m_2,prob=probinput,N_bins=N_bins_1d,histrange=None,x_lim=m2_lim,y_lim=None)

#histplot2d_part(axarr[2,1],m_2,z_1,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=m2_lim,y_lim=z1_lim)

#histplot2d_part(axarr[3,1],m_2,z_2,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=m2_lim,y_lim=z2_lim)

#histplot2d_part(axarr[4,1],m_2,d_proj,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=m2_lim,y_lim=dproj_lim)

## Third Column
#histplot1d_part(axarr[2,2],z_1,prob=probinput,N_bins=N_bins_1d,histrange=None,x_lim=z1_lim,y_lim=None)

#histplot2d_part(axarr[3,2],z_1,z_2,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=z1_lim,y_lim=z2_lim)

#histplot2d_part(axarr[4,2],z_1,d_proj,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=z1_lim,y_lim=dproj_lim)

## Fourth Column
#histplot1d_part(axarr[3,3],z_2,prob=probinput,N_bins=N_bins_1d,histrange=None,x_lim=z2_lim,y_lim=None)

#histplot2d_part(axarr[4,3],z_2,d_proj,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=z2_lim,y_lim=dproj_lim)

## Fifth Column
#histplot1d_part(axarr[4,4],d_proj,prob=probinput,N_bins=N_bins_1d,histrange=None,x_lim=dproj_lim,y_lim=None)

#filename = prefix+'_input-input.pdf'
#pylab.savefig(filename)

## Create input-geometry
#print 'creating input-geometry figure'
#f, axarr = pylab.subplots(3, 5)
#f.subplots_adjust(wspace=0, hspace=0)
#for i in numpy.arange(2):
    ## Remove the unecessary row axes labels
    #pylab.setp([a.get_xticklabels() for a in axarr[i,:]], visible=False)
#for i in numpy.arange(3):
    ## Remove the unecessary column axes labels
    #pylab.setp([a.get_yticklabels() for a in axarr[i,1:]], visible=False)

## rotate the redshift x-labels by 30 degrees so they don't bunch
#labels = axarr[2,2].get_xticklabels() 
#for label in labels: 
    #label.set_rotation(-90) 
#labels = axarr[2,3].get_xticklabels() 
#for label in labels: 
    #label.set_rotation(-90)
    
## Create the y-axes labels
#axarr[0,0].set_ylabel(r'$\alpha$ (degrees)',size=16)
#axarr[1,0].set_ylabel('$d_{3D}(t_{obs})$ (Mpc)',size=16)
#axarr[2,0].set_ylabel('$d_{max}$ (Mpc)',size=16)
## Create the x-axes labels
#axarr[2,0].set_xlabel('$M_{1} (10^{14} M_\odot)$',size=16)
#axarr[2,1].set_xlabel('$M_{2} (10^{14} M_\odot)$',size=16)
#axarr[2,2].set_xlabel('$z_1$',size=16)
#axarr[2,3].set_xlabel('$z_2$',size=16)
#axarr[2,4].set_xlabel('$d_{proj}(t_{obs})$ (Mpc)',size=16)

## First Column
#histplot2d_part(axarr[0,0],m_1,alpha,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=m1_lim,y_lim=alpha_lim)

#histplot2d_part(axarr[1,0],m_1,d_3d,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=m1_lim,y_lim=d3d_lim)

#histplot2d_part(axarr[2,0],m_1,d_max,prob=probinput,N_bins=N_bins_2d,histrange=(numpy.min(m_1),numpy.max(m_1),histrange_dmax[0],histrange_dmax[1]),x_lim=m1_lim,y_lim=dmax_lim)

## Second Column
#histplot2d_part(axarr[0,1],m_2,alpha,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=m2_lim,y_lim=alpha_lim)

#histplot2d_part(axarr[1,1],m_2,d_3d,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=m2_lim,y_lim=d3d_lim)

#histplot2d_part(axarr[2,1],m_2,d_max,prob=probinput,N_bins=N_bins_2d,histrange=(numpy.min(m_2),numpy.max(m_2),histrange_dmax[0],histrange_dmax[1]),x_lim=m2_lim,y_lim=dmax_lim)

## Third Column
#histplot2d_part(axarr[0,2],z_1,alpha,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=z1_lim,y_lim=alpha_lim)

#histplot2d_part(axarr[1,2],z_1,d_3d,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=z1_lim,y_lim=d3d_lim)

#histplot2d_part(axarr[2,2],z_1,d_max,prob=probinput,N_bins=N_bins_2d,histrange=(numpy.min(z_1),numpy.max(z_1),histrange_dmax[0],histrange_dmax[1]),x_lim=z1_lim,y_lim=dmax_lim)

## Fourth Column
#histplot2d_part(axarr[0,3],z_2,alpha,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=z2_lim,y_lim=alpha_lim)

#histplot2d_part(axarr[1,3],z_2,d_3d,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=z2_lim,y_lim=d3d_lim)

#histplot2d_part(axarr[2,3],z_2,d_max,prob=probinput,N_bins=N_bins_2d,histrange=(numpy.min(z_2),numpy.max(z_2),histrange_dmax[0],histrange_dmax[1]),x_lim=z2_lim,y_lim=dmax_lim)

## Fifth Column
#histplot2d_part(axarr[0,4],d_proj,alpha,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=dproj_lim,y_lim=alpha_lim)

#histplot2d_part(axarr[1,4],d_proj,d_3d,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=dproj_lim,y_lim=d3d_lim)

#histplot2d_part(axarr[2,4],d_proj,d_max,prob=probinput,N_bins=N_bins_2d,histrange=(numpy.min(d_proj),numpy.max(d_proj),histrange_dmax[0],histrange_dmax[1]),x_lim=dproj_lim,y_lim=dmax_lim)

#filename = prefix+'_input-geometry.pdf'
#pylab.savefig(filename)

## Create input-vt figure
#print 'creating input-vt figure'
#f, axarr = pylab.subplots(5, 5)
#f.subplots_adjust(wspace=0, hspace=0)
#for i in numpy.arange(4):
    ## Remove the unecessary row axes labels
    #pylab.setp([a.get_xticklabels() for a in axarr[i,:]], visible=False)
#for i in numpy.arange(5):
    ## Remove the unecessary column axes labels
    #pylab.setp([a.get_yticklabels() for a in axarr[i,1:]], visible=False)

## rotate the redshift x-labels by 30 degrees so they don't bunch
#labels = axarr[4,2].get_xticklabels() 
#for label in labels: 
    #label.set_rotation(-90) 
#labels = axarr[4,3].get_xticklabels() 
#for label in labels: 
    #label.set_rotation(-90)
    
## Create the y-axes labels
#axarr[0,0].set_ylabel('$v_{3D}(t_{obs})$ (km s$^{-1}$)',size=16)
#axarr[1,0].set_ylabel('$v_{3D}(t_{col})$ (km s$^{-1}$)',size=16)
#axarr[2,0].set_ylabel('TSC$_0$ (Gyr)',size=16)
#axarr[3,0].set_ylabel('TSC$_1$ (Gyr)',size=16)
#axarr[4,0].set_ylabel('T (Gyr)',size=16)
## Create the x-axes labels
#axarr[4,0].set_xlabel('$M_{1} (10^{14} M_\odot)$',size=16)
#axarr[4,1].set_xlabel('$M_{2} (10^{14} M_\odot)$',size=16)
#axarr[4,2].set_xlabel('$z_1$',size=16)
#axarr[4,3].set_xlabel('$z_2$',size=16)
#axarr[4,4].set_xlabel('$d_{proj}(t_{obs})$',size=16)

## First Column
#histplot2d_part(axarr[0,0],m_1,v_3d_obs,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=m1_lim,y_lim=v3dobs_lim)

#histplot2d_part(axarr[1,0],m_1,v_3d_col,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=m1_lim,y_lim=v3dcol_lim)

#histplot2d_part(axarr[2,0],m_1,TSM_0,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=m1_lim,y_lim=tsc0_lim)

#histplot2d_part(axarr[3,0],m_1[mask_TSM_1],TSM_1[mask_TSM_1],prob=probinput[mask_TSM_1],N_bins=N_bins_2d,histrange=None,x_lim=m1_lim,y_lim=tsc1_lim)

#histplot2d_part(axarr[4,0],m_1,T,prob=probinput,N_bins=N_bins_2d,histrange=(numpy.min(m_1),numpy.max(m_1),histrange_T[0],histrange_T[1]),x_lim=m1_lim,y_lim=t_lim)

## Second Column
#histplot2d_part(axarr[0,1],m_2,v_3d_obs,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=m2_lim,y_lim=v3dobs_lim)

#histplot2d_part(axarr[1,1],m_2,v_3d_col,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=m2_lim,y_lim=v3dcol_lim)

#histplot2d_part(axarr[2,1],m_2,TSM_0,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=m2_lim,y_lim=tsc0_lim)

#histplot2d_part(axarr[3,1],m_2[mask_TSM_1],TSM_1[mask_TSM_1],prob=probinput[mask_TSM_1],N_bins=N_bins_2d,histrange=None,x_lim=m2_lim,y_lim=tsc1_lim)

#histplot2d_part(axarr[4,1],m_2,T,prob=probinput,N_bins=N_bins_2d,histrange=(numpy.min(m_2),numpy.max(m_2),histrange_T[0],histrange_T[1]),x_lim=m2_lim,y_lim=t_lim)

## Third Column
#histplot2d_part(axarr[0,2],z_1,v_3d_obs,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=z1_lim,y_lim=v3dobs_lim)

#histplot2d_part(axarr[1,2],z_1,v_3d_col,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=z1_lim,y_lim=v3dcol_lim)

#histplot2d_part(axarr[2,2],z_1,TSM_0,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=z1_lim,y_lim=tsc0_lim)

#histplot2d_part(axarr[3,2],z_1[mask_TSM_1],TSM_1[mask_TSM_1],prob=probinput[mask_TSM_1],N_bins=N_bins_2d,histrange=None,x_lim=z1_lim,y_lim=tsc1_lim)

#histplot2d_part(axarr[4,2],z_1,T,prob=probinput,N_bins=N_bins_2d,histrange=(numpy.min(z_1),numpy.max(z_1),histrange_T[0],histrange_T[1]),x_lim=z1_lim,y_lim=t_lim)

## Fourth Column
#histplot2d_part(axarr[0,3],z_2,v_3d_obs,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=z2_lim,y_lim=v3dobs_lim)

#histplot2d_part(axarr[1,3],z_2,v_3d_col,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=z2_lim,y_lim=v3dcol_lim)

#histplot2d_part(axarr[2,3],z_2,TSM_0,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=z2_lim,y_lim=tsc0_lim)

#histplot2d_part(axarr[3,3],z_2[mask_TSM_1],TSM_1[mask_TSM_1],prob=probinput[mask_TSM_1],N_bins=N_bins_2d,histrange=None,x_lim=z2_lim,y_lim=tsc1_lim)

#histplot2d_part(axarr[4,3],z_2,T,prob=probinput,N_bins=N_bins_2d,histrange=(numpy.min(z_2),numpy.max(z_2),histrange_T[0],histrange_T[1]),x_lim=z2_lim,y_lim=t_lim)

## Fifth Column
#histplot2d_part(axarr[0,4],d_proj,v_3d_obs,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=dproj_lim,y_lim=v3dobs_lim)

#histplot2d_part(axarr[1,4],d_proj,v_3d_col,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=dproj_lim,y_lim=v3dcol_lim)

#histplot2d_part(axarr[2,4],d_proj,TSM_0,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=dproj_lim,y_lim=tsc0_lim)

#histplot2d_part(axarr[3,4],d_proj[mask_TSM_1],TSM_1[mask_TSM_1],prob=probinput[mask_TSM_1],N_bins=N_bins_2d,histrange=None,x_lim=dproj_lim,y_lim=tsc1_lim)

#histplot2d_part(axarr[4,4],d_proj,T,prob=probinput,N_bins=N_bins_2d,histrange=(numpy.min(d_proj),numpy.max(d_proj),histrange_T[0],histrange_T[1]),x_lim=dproj_lim,y_lim=t_lim)

#filename = prefix+'_input-vt.pdf'
#pylab.savefig(filename)

## Create geometry-geometry figure
#print 'creating geometry-geometry figure'
#f, axarr = pylab.subplots(3, 3)
#f.subplots_adjust(wspace=0, hspace=0)
## Remove the [0,0] plot's y tickmarks
#pylab.setp([axarr[0,0].get_yticklabels()], visible=False)
#for i in numpy.arange(2):
    ## Remove unnecessary subplots
    #pylab.setp([a.get_axes() for a in axarr[i,i+1:]], visible=False)
    ## Remove the unecessary row axes labels
    #pylab.setp([a.get_xticklabels() for a in axarr[i,:]], visible=False)
    ## Remove the unecessary column axes labels
    #pylab.setp([a.get_yticklabels() for a in axarr[i+1,1:]], visible=False)
    
## Create the y-axes labels
##axarr[0,0].set_ylabel(r'$\alpha$ (degrees)')
#axarr[1,0].set_ylabel('$d_{3D}(t_{obs})$ (Mpc)',size=16)
#axarr[2,0].set_ylabel('$d_{max}$ (Mpc)',size=16)
## Create the x-axes labels
#axarr[2,0].set_xlabel(r'$\alpha$ (degrees)',size=16)
#axarr[2,1].set_xlabel('$d_{3D}(t_{obs})$ (Mpc)',size=16)
#axarr[2,2].set_xlabel('$d_{max}$ (Mpc)',size=16)

## First Column
#histplot1d_part(axarr[0,0],alpha,prob=probinput,N_bins=N_bins_1d,histrange=None,x_lim=alpha_lim,y_lim=None)

#histplot2d_part(axarr[1,0],alpha,d_3d,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=alpha_lim,y_lim=d3d_lim)

#histplot2d_part(axarr[2,0],alpha,d_max,prob=probinput,N_bins=N_bins_2d,histrange=(numpy.min(alpha),numpy.max(alpha),histrange_dmax[0],histrange_dmax[1]),x_lim=alpha_lim,y_lim=dmax_lim)

## Second Column
#histplot1d_part(axarr[1,1],d_3d,prob=probinput,N_bins=N_bins_1d,histrange=None,x_lim=d3d_lim,y_lim=None)

#histplot2d_part(axarr[2,1],d_3d,d_max,prob=probinput,N_bins=N_bins_2d,histrange=(numpy.min(d_3d),numpy.max(d_3d),histrange_dmax[0],histrange_dmax[1]),x_lim=d3d_lim,y_lim=dmax_lim)

## Third Column
#histplot1d_part(axarr[2,2],d_max,prob=probinput,N_bins=N_bins_1d,histrange=histrange_dmax,x_lim=dmax_lim,y_lim=None)

#filename = prefix+'_geometry-geometry.pdf'
#pylab.savefig(filename)

## create the geometry-vt figure
#print 'creating geometry-vt figure'
#f, axarr = pylab.subplots(5, 3)
#f.subplots_adjust(wspace=0, hspace=0)
#for i in numpy.arange(4):
    ## Remove the unecessary row axes labels
    #pylab.setp([a.get_xticklabels() for a in axarr[i,:]], visible=False)
#for i in numpy.arange(5):
    ## Remove the unecessary column axes labels
    #pylab.setp([a.get_yticklabels() for a in axarr[i,1:]], visible=False)
    
## Create the y-axes labels
#axarr[0,0].set_ylabel('$v_{3D}(t_{obs})$ (km s$^{-1}$)',size=16)
#axarr[1,0].set_ylabel('$v_{3D}(t_{col})$ (km s$^{-1}$)',size=16)
#axarr[2,0].set_ylabel('TSC$_0$ (Gyr)',size=16)
#axarr[3,0].set_ylabel('TSC$_1$ (Gyr)',size=16)
#axarr[4,0].set_ylabel('T (Gyr)',size=16)
## Create the x-axes labels
#axarr[4,0].set_xlabel(r'$\alpha$ (degrees)',size=16)
#axarr[4,1].set_xlabel('$d_{3D}(t_{obs})$ (Mpc)',size=16)
#axarr[4,2].set_xlabel('$d_{max}$ (Mpc)',size=16)

## First Column
#histplot2d_part(axarr[0,0],alpha,v_3d_obs,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=alpha_lim,y_lim=v3dobs_lim)

#histplot2d_part(axarr[1,0],alpha,v_3d_col,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=alpha_lim,y_lim=v3dcol_lim)

#histplot2d_part(axarr[2,0],alpha,TSM_0,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=alpha_lim,y_lim=tsc0_lim)

#histplot2d_part(axarr[3,0],alpha[mask_TSM_1],TSM_1[mask_TSM_1],prob=probinput[mask_TSM_1],N_bins=N_bins_2d,histrange=None,x_lim=alpha_lim,y_lim=tsc1_lim)

#histplot2d_part(axarr[4,0],alpha,T,prob=probinput,N_bins=N_bins_2d,histrange=(numpy.min(alpha),numpy.max(alpha),histrange_T[0],histrange_T[1]),x_lim=alpha_lim,y_lim=t_lim)

## Second Column
#histplot2d_part(axarr[0,1],d_3d,v_3d_obs,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=d3d_lim,y_lim=v3dobs_lim)

#histplot2d_part(axarr[1,1],d_3d,v_3d_col,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=d3d_lim,y_lim=v3dcol_lim)

#histplot2d_part(axarr[2,1],d_3d,TSM_0,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=d3d_lim,y_lim=tsc0_lim)

#histplot2d_part(axarr[3,1],d_3d[mask_TSM_1],TSM_1[mask_TSM_1],prob=probinput[mask_TSM_1],N_bins=N_bins_2d,histrange=None,x_lim=d3d_lim,y_lim=tsc1_lim)

#histplot2d_part(axarr[4,1],d_3d,T,prob=probinput,N_bins=N_bins_2d,histrange=(numpy.min(d_3d),numpy.max(d_3d),histrange_T[0],histrange_T[1]),x_lim=d3d_lim,y_lim=t_lim)

## Third Column
#histplot2d_part(axarr[0,2],d_max,v_3d_obs,prob=probinput,N_bins=N_bins_2d,histrange=(histrange_dmax[0],histrange_dmax[1],numpy.min(v_3d_obs),numpy.max(v_3d_obs)),x_lim=dmax_lim,y_lim=v3dobs_lim)

#histplot2d_part(axarr[1,2],d_max,v_3d_col,prob=probinput,N_bins=N_bins_2d,histrange=(histrange_dmax[0],histrange_dmax[1],numpy.min(v_3d_col),numpy.max(v_3d_col)),x_lim=dmax_lim,y_lim=v3dcol_lim)

#histplot2d_part(axarr[2,2],d_max,TSM_0,prob=probinput,N_bins=N_bins_2d,histrange=(histrange_dmax[0],histrange_dmax[1],numpy.min(TSM_0),numpy.max(TSM_0)),x_lim=dmax_lim,y_lim=tsc0_lim)

#histplot2d_part(axarr[3,2],d_max[mask_TSM_1],TSM_1[mask_TSM_1],prob=probinput[mask_TSM_1],N_bins=N_bins_2d,histrange=(histrange_dmax[0],histrange_dmax[1],numpy.min(TSM_1[mask_TSM_1]),numpy.max(TSM_1[mask_TSM_1])),x_lim=dmax_lim,y_lim=tsc1_lim)

#histplot2d_part(axarr[4,2],d_max,T,prob=probinput,N_bins=N_bins_2d,histrange=(histrange_dmax[0],histrange_dmax[1],histrange_T[0],histrange_T[1]),x_lim=dmax_lim,y_lim=t_lim)

#filename = prefix+'_geometry-vt.pdf'
#pylab.savefig(filename)

## create vt-vt figure
#print 'creating vt-vt figure'
#f, axarr = pylab.subplots(5, 5)
#f.subplots_adjust(wspace=0, hspace=0)
## Remove the [0,0] plot's y tickmarks
#pylab.setp([axarr[0,0].get_yticklabels()], visible=False)
#for i in numpy.arange(4):
    ## Remove unnecessary subplots
    #pylab.setp([a.get_axes() for a in axarr[i,i+1:]], visible=False)
    ## Remove the unecessary row axes labels
    #pylab.setp([a.get_xticklabels() for a in axarr[i,:]], visible=False)
    ## Remove the unecessary column axes labels
    #pylab.setp([a.get_yticklabels() for a in axarr[i+1,1:]], visible=False)
    
## Create the y-axes labels
#axarr[1,0].set_ylabel('$v_{3D}(t_{col})$ (km s$^{-1}$)',size=16)
#axarr[2,0].set_ylabel('TSC$_0$ (Gyr)',size=16)
#axarr[3,0].set_ylabel('TSC$_1$ (Gyr)',size=16)
#axarr[4,0].set_ylabel('T (Gyr)',size=16)
## Create the x-axes labels
#axarr[4,0].set_xlabel('$v_{3D}(t_{obs})$ (km s$^{-1}$)',size=16)
#axarr[4,1].set_xlabel('$v_{3D}(t_{col})$ (km s$^{-1}$)',size=16)
#axarr[4,2].set_xlabel('TSC$_0$ (Gyr)',size=16)
#axarr[4,3].set_xlabel('TSC$_1$ (Gyr)',size=16)
#axarr[4,4].set_xlabel('T (Gyr)',size=16)

## rotate the velocity x-labels by 30 degrees so they don't bunch
#labels = axarr[4,0].get_xticklabels() 
#for label in labels: 
    #label.set_rotation(-90) 
#labels = axarr[4,1].get_xticklabels() 
#for label in labels: 
    #label.set_rotation(-90)

## First Column
#histplot1d_part(axarr[0,0],v_3d_obs,prob=probinput,N_bins=N_bins_1d,histrange=None,x_lim=v3dobs_lim,y_lim=None)

#histplot2d_part(axarr[1,0],v_3d_obs,v_3d_col,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=v3dobs_lim,y_lim=v3dcol_lim)

#histplot2d_part(axarr[2,0],v_3d_obs,TSM_0,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=v3dobs_lim,y_lim=tsc0_lim)

#histplot2d_part(axarr[3,0],v_3d_obs[mask_TSM_1],TSM_1[mask_TSM_1],prob=probinput[mask_TSM_1],N_bins=N_bins_2d,histrange=None,x_lim=v3dobs_lim,y_lim=tsc1_lim)

#histplot2d_part(axarr[4,0],v_3d_obs,T,prob=probinput,N_bins=N_bins_2d,histrange=(numpy.min(v_3d_obs),numpy.max(v_3d_obs),histrange_T[0],histrange_T[1]),x_lim=v3dobs_lim,y_lim=t_lim)

## Second Column
#histplot1d_part(axarr[1,1],v_3d_col,prob=probinput,N_bins=N_bins_1d,histrange=None,x_lim=v3dcol_lim,y_lim=None)

#histplot2d_part(axarr[2,1],v_3d_col,TSM_0,prob=probinput,N_bins=N_bins_2d,histrange=None,x_lim=v3dcol_lim,y_lim=tsc0_lim)

#histplot2d_part(axarr[3,1],v_3d_col[mask_TSM_1],TSM_1[mask_TSM_1],prob=probinput[mask_TSM_1],N_bins=N_bins_2d,histrange=None,x_lim=v3dcol_lim,y_lim=tsc1_lim)

#histplot2d_part(axarr[4,1],v_3d_col,T,prob=probinput,N_bins=N_bins_2d,histrange=(numpy.min(v_3d_col),numpy.max(v_3d_col),histrange_T[0],histrange_T[1]),x_lim=v3dcol_lim,y_lim=t_lim)

## Third Column
#histplot1d_part(axarr[2,2],TSM_0,prob=probinput,N_bins=N_bins_1d,histrange=None,x_lim=tsc0_lim,y_lim=None)

#histplot2d_part(axarr[3,2],TSM_0[mask_TSM_1],TSM_1[mask_TSM_1],prob=probinput[mask_TSM_1],N_bins=N_bins_2d,histrange=None,x_lim=tsc0_lim,y_lim=tsc1_lim)

#histplot2d_part(axarr[4,2],TSM_0,T,prob=probinput,N_bins=N_bins_2d,histrange=(numpy.min(TSM_0),numpy.max(TSM_0),histrange_T[0],histrange_T[1]),x_lim=tsc0_lim,y_lim=t_lim)

## Fourth Column
#histplot1d_part(axarr[3,3],TSM_1[mask_TSM_1],prob=probinput[mask_TSM_1],N_bins=N_bins_1d,histrange=None,x_lim=tsc1_lim,y_lim=None)

#histplot2d_part(axarr[4,3],TSM_1[mask_TSM_1],T[mask_TSM_1],prob=probinput[mask_TSM_1],N_bins=N_bins_2d,histrange=(numpy.min(TSM_1[mask_TSM_1]),numpy.max(TSM_1[mask_TSM_1]),histrange_T[0],histrange_T[1]),x_lim=tsc1_lim,y_lim=t_lim)

## Fifth Column
#histplot1d_part(axarr[4,4],T,prob=probinput,N_bins=N_bins_1d,histrange=histrange_T,x_lim=t_lim,y_lim=None)

#filename = prefix+'_vt-vt.pdf'
#pylab.savefig(filename)





##m1 m2
#histplot2d(m_1,m_2,'m1m2',prob=prob,N_bins=N_bins_2d,histrange=None,x_lim=None,y_lim=None,x_label='m1',y_label='m2')

###v_col TSC_0
#histplot2dTSC(v_3d_col,TSM_0,'vcolTSC0',prob=prob,N_bins=N_bins_2d,histrange=(1000,4000,0,5),x_lim=(1000,4200),y_lim=(0,3),x_label='Relative Collision Velocity ($km s^{-1}$)',y_label='Time Since Collision ($Gyr$)')

### Create the 1D histograms
#histplot1d(m_1,'m1',prob=probinput,N_bins=N_bins_1d,histrange=None,x_lim=None,y_lim=None,x_label='m1',y_label=ylab,legend=None)

#histplot1d(m_2,'m2',prob=probinput,N_bins=N_bins_1d,histrange=None,x_lim=None,y_lim=None,x_label='m2',y_label=ylab,legend=None)

#histplot1d(z_1,'z1',prob=probinput,N_bins=N_bins_1d,histrange=None,x_lim=None,y_lim=None,x_label='z1',y_label=ylab,legend=None)

#histplot1d(z_2,'z2',prob=probinput,N_bins=N_bins_1d,histrange=None,x_lim=None,y_lim=None,x_label='z2',y_label=ylab,legend=None)

#histplot1d(d_proj,'dproj',prob=probinput,N_bins=N_bins_1d,histrange=None,x_lim=None,y_lim=None,x_label='dproj',y_label=ylab,legend=None)

#histplot1d(alpha,'alpha',prob=probinput,N_bins=N_bins_1d,histrange=None,x_lim=None,y_lim=None,x_label='alpha',y_label=ylab,legend=None)

#histplot1d(d_3d,'d3d',prob=probinput,N_bins=N_bins_1d,histrange=None,x_lim=None,y_lim=None,x_label='d3d',y_label=ylab,legend=None)

#histplot1d(d_max,'dmax',prob=probinput,N_bins=N_bins_1d,histrange=(0,10),x_lim=None,y_lim=None,x_label='dmax',y_label=ylab,legend=None)

#histplot1d(v_3d_obs,'v3dobs',prob=probinput,N_bins=N_bins_1d,histrange=None,x_lim=None,y_lim=None,x_label='v_3D_obs',y_label=ylab,legend=None)

#histplot1d(v_3d_col,'v3dcol',prob=probinput,N_bins=N_bins_1d,histrange=None,x_lim=None,y_lim=None,x_label='v_3D_col',y_label=ylab,legend=None)

#histplot1d(TSM_0,'TSC_0',prob=probinput,N_bins=N_bins_1d,histrange=None,x_lim=None,y_lim=None,x_label='TSC_0',y_label=ylab,legend=None)

#histplot1d(TSM_1[mask_TSM_1],'TSC_1',prob=probinput[mask_TSM_1],N_bins=N_bins_1d,histrange=None,x_lim=None,y_lim=None,x_label='TSC_1',y_label=ylab,legend=None)

#histplot1d(T,'T',prob=probinput,N_bins=N_bins_1d,histrange=(0,30),x_lim=None,y_lim=None,x_label='T',y_label=ylab,legend=None)

pylab.show()
