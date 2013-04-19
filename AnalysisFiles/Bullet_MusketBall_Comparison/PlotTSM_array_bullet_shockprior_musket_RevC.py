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
prefix = '/Users/dawson/Git/merging-cluster-dynamics-paper/AnalysisFiles/BulletCluster/DefaultPriors/bulletrun_'
index = ('0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19')
prefix_output = 'bulletrun_shockprior'
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

#Filter the arrays by the time prior
# Shock TSC_0 prior
tsc0_prior = 1 #Gyr; loose cut to get rid of near zero tail, just for formating concerns
a = 0.2
b = 0.2

mask_TSM_0 = TSM_0 < tsc0_prior
m_1 = m_1[mask_TSM_0]
m_2 = m_2[mask_TSM_0]
z_1 = z_1[mask_TSM_0]
z_2 = z_2[mask_TSM_0]
d_proj = d_proj[mask_TSM_0]
v_rad_obs = v_rad_obs[mask_TSM_0]
alpha = alpha[mask_TSM_0]
v_3d_obs = v_3d_obs[mask_TSM_0]
d_3d = d_3d[mask_TSM_0]
v_3d_col = v_3d_col[mask_TSM_0]
d_max = d_max[mask_TSM_0]
TSM_0 = TSM_0[mask_TSM_0]
TSM_1 = TSM_1[mask_TSM_0]
T = T[mask_TSM_0]
prob = prob[mask_TSM_0]

# add new prob
prob_shock = 0.5-0.5*numpy.tanh((TSM_0-a)/b)
prob *= prob_shock

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
    if y_lim != None:
        pylab.ylim((y_lim[0],y_lim[1]))
    for i in numpy.arange(N_bins):
        if i == 0:
            x_binned = numpy.ones(hist[i])*(binedges[i]+binedges[i+1])/2
        else:
            x_temp = numpy.ones(hist[i])*(binedges[i]+binedges[i+1])/2
            x_binned = numpy.concatenate((x_binned,x_temp))
    loc = biweightLoc(x_binned)
    ll_68, ul_68 = bcpcl(loc,x_binned,1)
    ll_95, ul_95 = bcpcl(loc,x_binned,2)

    mask_68 = numpy.logical_and(binedges>=ll_68,binedges<=ul_68)
    mask_95 = numpy.logical_and(binedges>=ll_95,binedges<=ul_95)

    pylab.hist(x,bins=binedges[mask_95],weights=prob,histtype='stepfilled',linewidth=0,color=(158/255.,202/255.,225/255.))
    pylab.hist(x,bins=binedges[mask_68],weights=prob,histtype='stepfilled',linewidth=0,color=(49/255.,130/255.,189/255.))
    
    # determine the height of the histogram at the loc
    #determine the binedge to the left and right of the location
    idx = tools.argnearest(binedges,loc)
    near = tools.nearest(binedges,loc)
    if loc >= near:
        y_loc = hist[idx]
    else:
        y_loc = hist[idx-1]
    
    # Create location and confidence interval line plots
    pylab.plot((loc,loc),(pylab.ylim()[0],y_loc),'--k',linewidth=2,label='$C_{BI}$')
    #pylab.plot((ll_68,ll_68),(pylab.ylim()[0],pylab.ylim()[1]),'-.',linewidth=2,color='#800000',label='68% $IC_{B_{BI}}$')
    #pylab.plot((ul_68,ul_68),(pylab.ylim()[0],pylab.ylim()[1]),'-.',linewidth=2,color='#800000')
    #pylab.plot((ll_95,ll_95),(pylab.ylim()[0],pylab.ylim()[1]),':',linewidth=2,color='#0000A0',label='95% $IC_{B_{BI}}$')
    #pylab.plot((ul_95,ul_95),(pylab.ylim()[0],pylab.ylim()[1]),':',linewidth=2,color='#0000A0')
    
    # replot the 1D histogram so that the line is not covered by colored sections
    hist, binedges, tmp = pylab.hist(x,bins=N_bins,histtype='step',weights=prob,range=histrange,color='k',linewidth=2)
    
    if x_label != None:
        pylab.xlabel(x_label,fontsize=16)
    if y_label != None:    
        pylab.ylabel(y_label,fontsize=16)
    if x_lim != None:
        pylab.xlim(x_lim)
    if y_lim != None:
        pylab.ylim(y_lim)
    if legend != None:
        pylab.legend()
    fontsize=14
    ax = pylab.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    filename = prefix+'_histplot1D.pdf'
    pylab.savefig(filename)
    
    print '{0}, {1:0.4f}, {2:0.4f}, {3:0.4f}, {4:0.4f}, {5:0.4f}'.format(prefix,loc,ll_68,ul_68,ll_95,ul_95)
    
    return loc, ll_68, ul_68, ll_95, ul_95

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
    
    # Create the color approximate 68% and 95% color regions
    mask_68 = numpy.logical_and(binedges>=ll_68,binedges<=ul_68)
    mask_95 = numpy.logical_and(binedges>=ll_95,binedges<=ul_95)
    
    ax.hist(x,bins=binedges[mask_68],weights=prob,histtype='stepfilled',color=(49/255.,130/255.,189/255.))
    ax.hist(x,bins=binedges[mask_95],weights=prob,histtype='stepfilled',color=(158/255.,202/255.,225/255.))
    

    # Create location and confidence interval line plots
    ax.plot((loc,loc),(ax.get_ylim()[0],ax.get_ylim()[1]),'--k',linewidth=1,label='$C_{BI}$')
    #ax.plot((ll_68,ll_68),(ax.get_ylim()[0],ax.get_ylim()[1]),'-.',linewidth=1,color='#800000',label='68% $IC_{B_{BI}}$')
    #ax.plot((ul_68,ul_68),(ax.get_ylim()[0],ax.get_ylim()[1]),'-.',linewidth=1,color='#800000')
    #ax.plot((ll_95,ll_95),(ax.get_ylim()[0],ax.get_ylim()[1]),':',linewidth=1,color='#0000A0',label='95% $IC_{B_{BI}}$')
    #ax.plot((ul_95,ul_95),(ax.get_ylim()[0],ax.get_ylim()[1]),':',linewidth=1,color='#0000A0')
    
    # replot the 1D histogram so that the line is not covered by colored sections
    hist, binedges, tmp = pylab.hist(x,bins=N_bins,histtype='step',weights=prob,range=histrange,color='k',linewidth=2)    
    
    if x_lim != None:
        ax.set_xlim(x_lim)
    if y_lim != None:
        ax.set_ylim(y_lim)
    return loc, ll_68, ul_68, ll_95, ul_95


def histplot2d_part(ax,x,y,prob=None,N_bins=100,histrange=None,x_lim=None,y_lim=None,scatter=None,gray=False):
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
    if gray:
        CS = ax.contour(X,Y,H,(h_2sigma,h_1sigma),linewidths=(2,2),linestyles='dashed',colors='0.5')
    else:
        CS = ax.contour(X,Y,H,(h_2sigma,h_1sigma),linewidths=(2,2),colors=((158/255.,202/255.,225/255.),(49/255.,130/255.,189/255.))) 
    # imshow
    #im = ax.imshow(H,cmap=ax.cm.gray)
    if scatter != None:
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
    #pylab.scatter(v_macs,t_macs,s=140, c='0.4',markeredgecolor='0.4', marker='^',label='MACS J0025.4')
    #pylab.scatter(v_a520,t_a520,s=140,c='0.4',markeredgecolor='0.4',marker='o',label='A520')
    #pylab.scatter(v_pandora,t_pandora,s=140,c='0.4',markeredgecolor='0.4',marker='p',label='A2744')
    
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
    
    filename = prefix+'_histplot2dTSC.pdf'
    pylab.savefig(filename)
    
    return fig

###############################################
# Read in the Musket Ball Cluster properties

#prefix from TSM_MCcalc.py
prefix = '/Users/dawson/Git/merging-cluster-dynamics-paper/AnalysisFiles/MusketBallCluster/run_v11_'
index = ('0','1','2','3','4','5','6','7','8','9')
#Histogram bins
N_bins = 100
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
mm_1 = loadcombo(prefix,index,'m_1')
mm_2 = loadcombo(prefix,index,'m_2')
mz_1 = loadcombo(prefix,index,'z_1')
mz_2 = loadcombo(prefix,index,'z_2')
md_proj = loadcombo(prefix,index,'d_proj')
mv_rad_obs = loadcombo(prefix,index,'v_rad_obs')
malpha = loadcombo(prefix,index,'alpha')
mv_3d_obs = loadcombo(prefix,index,'v_3d_obs')
md_3d = loadcombo(prefix,index,'d_3d')
mv_3d_col = loadcombo(prefix,index,'v_3d_col')
md_max = loadcombo(prefix,index,'d_max')
mTSM_0 = loadcombo(prefix,index,'TSM_0')
mTSM_1 = loadcombo(prefix,index,'TSM_1')
mT = loadcombo(prefix,index,'T')
mprob = loadcombo(prefix,index,'prob')

# for some reason 4 of the T, and thus prob, elements have nan values. Need to
# remove these cases from all arrays
mask_nan = ~numpy.isnan(mT)
mm_1 = mm_1[mask_nan]
mm_2 = mm_2[mask_nan]
mz_1 = mz_1[mask_nan]
mz_2 = mz_2[mask_nan]
md_proj = md_proj[mask_nan]
mv_rad_obs = mv_rad_obs[mask_nan]
malpha = malpha[mask_nan]
mv_3d_obs = mv_3d_obs[mask_nan]
md_3d = md_3d[mask_nan]
mv_3d_col = mv_3d_col[mask_nan]
md_max = md_max[mask_nan]
mTSM_0 = mTSM_0[mask_nan]
mTSM_1 = mTSM_1[mask_nan]
mT = mT[mask_nan]
mprob = mprob[mask_nan]


################################################
f = 10
rand = numpy.random.randint(numpy.size(TSM_0),size=f*numpy.max((numpy.size(TSM_0),numpy.size(mTSM_0))))
mrand = numpy.random.randint(numpy.size(mTSM_0),size=f*numpy.max((numpy.size(TSM_0),numpy.size(mTSM_0))))

###############################################

histplot1d(mTSM_0[mrand]/TSM_0[rand],'TSCcomparison_revB',prob=mprob[mrand]*prob[rand],N_bins=400,histrange=(0,30),x_lim=(0,20),y_lim=None,x_label='$TSC_{0_{Musket}}/TSC_{0_{Bullet}}$',y_label='Number of Realizations',legend=None)

histplot1d(mTSM_0[mrand]-TSM_0[rand],'TSCsubtraction_revB',prob=mprob[mrand]*prob[rand],N_bins=400,histrange=None,x_lim=(-0.5,8),y_lim=None,x_label='$TSC_{0_{Musket}}-TSC_{0_{Bullet}}$ (Gyr)',y_label='Number of Realizations',legend=None)


fig = pylab.figure()
ax = fig.add_subplot(111)

histplot2d_part(ax,mv_3d_col,mTSM_0,prob=mprob,N_bins=N_bins_2d,histrange=(1500,3000,0,6),x_lim=(1500,4500),y_lim=(0,5.5),scatter=True)
#ax.legend(loc=0)

# add bullet cluster contours for reference
histplot2d_part(ax,v_3d_col,TSM_0,prob=prob,N_bins=N_bins_2d,histrange=(2000,4500,0,1),x_lim=(1500,4500),y_lim=(0,5.5),scatter=None,gray=True)
#ax.legend(loc=0)
fontsize=14
ax = pylab.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
pylab.xlabel('$v_{3D}(t_{col})$ (km s$^{-1}$)',size=16)
pylab.ylabel('$TSC_0$ (Gyr)',size=16)
pylab.savefig('MusketTSCwithBullet_revC.pdf')

#print 'Number of Bullet Cluster realizations = {0}'.format(numpy.size(TSM_0))
#print 'Number of Musket Ball Cluster realizations = {0}'.format(numpy.size(mTSM_0))

## Create the combined TSC_0 and TSC_1 plot
#fig = pylab.figure()
#histrange=None
##concatinate the TSC arrays
#TSM_combo = numpy.concatenate((TSM_0,TSM_1[mask_TSM_1]))
#prob_combo = numpy.concatenate((prob,prob[mask_TSM_1]))
#hist, binedges, tmp = pylab.hist(TSM_combo,bins=N_bins,histtype='step',weights=prob_combo,range=histrange,color='k',linewidth=2)  
#pylab.hist(TSM_0,bins=binedges,histtype='step',weights=prob,range=histrange,color='k',linewidth=2,linestyle='dashed')  
#pylab.hist(TSM_1[mask_TSM_1],bins=binedges,histtype='step',weights=prob[mask_TSM_1],range=histrange,color='k',linewidth=2,linestyle='dashdot')

#print 'fraction of total cases with valid TSC_1 = {0}'.format(numpy.size(TSM_1[mask_TSM_1])/numpy.size(TSM_0))

pylab.show()
