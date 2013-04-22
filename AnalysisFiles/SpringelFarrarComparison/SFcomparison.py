'''
The purpose of this code is to compare the Dawson (2012) merging cluster model
predictions with they Springel and Farrar (2007) results of hydrodynamics
simulations of the Bullet Cluster.

Since my model takes observed parameters as input (e.g. mass of the subclusters,
projected separation in the observed state, and relative velocity in the
observed state) I will treat their "observed state" as input and calculate my 
model's predictions of the Time-Since-Collision (TSC) and relative velocity 
throughout the merger (from collison state to observed state). The choice for
these tow parameters is because Springle and Farrar (2007) plot these
parameters in figure 4.

The code is reflective of MCMAC v0.1 (git://github.com/MCTwo/MCMAC.git).
'''
from __future__ import division
import numpy
import scipy.integrate
import pickle
import profiles
import time
import sys
import cosmo

### User Inputs
# mass of each subcluster
m_1 = 1.5e15
m_2 = 1.5e14
# concentration of each subcluster
c_1 = 7.2
c_2 = 2.0
# redshift of each subcluster
z_1 = 0.29560
z_2 = 0.29826
# 3D relative velocity of the subclusters in the "observed" state
v_3d_obs = 2700
# 3D separation of the subclusters in the "observed" state
d_3d = 0.62
# MCMAC analysis resolution parameters
del_mesh=200
TSM_mesh=400

# Constants and conversions
G = 4.3*10**(-9) #newton's constant in units of Mpc*(km/s)**2 / M_sun
c = 3e5 #speed of light km/s
sinGyr = 31556926.*10**9 # s in a Giga-year
kginMsun = 1.98892*10**30 # kg in a solar mass
kminMpc = 3.08568025*10**19 # km in a Megaparsec
minMpc = 3.08568025*10**22 # m in a Megaparsec            
r2d = 180/numpy.pi # radians to degrees conversion factor

### Functions

def NFWprop(M_200,z,c):
    '''
    Determines the NFW halo related properties. Added this for the case of user
    specified concentration.
    Input:
    M_200 = [float; units:M_sun] mass of the halo. Assumes M_200 with
        respect to the critical density at the halo redshift.    
    z = [float; unitless] redshift of the halo.
    c = [float; unitless] concentration of the NFW halo.
    '''
    # CONSTANTS
    rho_cr = cosmo.rhoCrit(z)/kginMsun*minMpc**3
    #calculate the r_200 radius
    r_200 = (M_200*3/(4*numpy.pi*200*rho_cr))**(1/3.)
    del_c = 200/3.*c**3/(numpy.log(1+c)-c/(1+c))
    r_s = r_200/c
    rho_s = del_c*rho_cr   
    return del_c, r_s, r_200, c, rho_s

def f(x,a,b):
    '''
    This is the functional form of the time since merger integral for two point
    masses.
    '''
    return 1/numpy.sqrt(a+b/x)

def TSMptpt(m_1,m_2,r_200_1,r_200_2,d_end,E):
    '''
    This function calculates the time it takes for the system to go from a
    separation of r_200_1+r_200_2 to d_end. It is based on the equations of
    motion of two point masses, which is valid in the regime where the 
    subclusters no longer overlap.
    Input:
    m_1 = [float; units: M_sun] mass of subcluster 1
    m_2 = [float; units: M_sun] mass of subcluster 2
    r_200_1 = [float; units: Mpc] NFW r_200 radius of subcluster 1
    r_200_2 = [float; units: Mpc] NFW r_200 radius of subcluster 2
    d_end = [float; units: Mpc] the final separation of interest
    E = [float; units: (km/s)^2*M_sun] the total energy (PE+KE) of the two subcluster system
    Output:
    t = [float; units: Gyr] the time it takes for the system to go from a
        separation of r_200_1+r_200_2 to d_end
    '''
    d_start = r_200_1+r_200_2
    C = G*m_1*m_2
    mu = m_1*m_2/(m_1+m_2)
    if E < 0:
        integral = scipy.integrate.quad(lambda x: f(x,E,C),d_start,d_end)[0]
        t = numpy.sqrt(mu/2)*integral/sinGyr*kminMpc        
    else:
        print 'TSMptpt: error total energy should not be > 0, exiting'
        sys.exit()
    if t < 0:
        print 'TSM < 0 encountered'    
    return t

def PEnfwnfw(d,m_1,rho_s_1,r_s_1,r_200_1,m_2,rho_s_2,r_s_2,r_200_2,N=100):
    '''
    This function calculates the potential energy of two truncated NFW halos.
    Input:
    d = [float; units:Mpc] center to center 3D separation of the subclusters
    m_1 = [float; units:M_sun] mass of subcluster 1 out to r_200
    rho_s_1 = [float; units:M_sun/Mpc^3] NFW scale density of subcluster 1
    r_s_1 = [float; units:Mpc] NFW scale radius of subcluster 1
    r_200_1 = [float; units:Mpc] r_200 of subcluster 1
    m_2 = [float; units:M_sun] mass of subcluster 2 out to r_200
    rho_s_2 = [float; units:M_sun/Mpc^3] NFW scale density of subcluster 2
    r_s_2 = [float; units:Mpc] NFW scale radius of subcluster 2
    r_200_2 = [float; units:Mpc] r_200 of subcluster 2
    N = [int] number of mass elements along one coordinate axis for numerical
        integration approximation
    Output:
    V_total = [float; units:(km/s)^2*M_sun] total potential energy of the two
        subcluster system
    '''
    if d >= r_200_1+r_200_2:
        V_total = -G*m_1*m_2/d
    else:
        # mass element sizes
        dr = r_200_2/N
        dt = numpy.pi/N
        i,j = numpy.meshgrid(numpy.arange(N),numpy.arange(N))
        # distance of 2nd NFW halo mass element from center of 1st NFW halo
        r = numpy.sqrt(((2*i+1)*dr/2*numpy.sin((2*j+1)*dt/2))**2+
                       (d+(2*i+1)*dr/2*numpy.cos((2*j+1)*dt/2))**2)
        #mass elements of 2nd NFW halo  
        m = 2*numpy.pi*rho_s_2*r_s_2**3*(numpy.cos(j*dt)-numpy.cos((j+1)*dt))*(1/(1+(i+1)*dr/r_s_2)-1/(1+i*dr/r_s_2)+numpy.log(((i+1)*dr+r_s_2)/(i*dr+r_s_2)))
        #determine if 2nd halo mass element is inside or outside of 1st halo
        mask_gt = r>=r_200_1
        mask_lt = r<r_200_1
        # potential energy of each mass element (km/s)^2 * M_sun
        # NFW PE
        V_nfw = numpy.sum(-4*numpy.pi*G*rho_s_1*r_s_1**3/(r[mask_lt]+dr)*(numpy.log(1+(r[mask_lt]+dr)/r_s_1)-(r[mask_lt]+dr)/(r_s_1+r_200_1))*m[mask_lt])
        # Point mass PE
        V_pt = numpy.sum(-G*m_1*m[mask_gt]/r[mask_gt])
        V_total = V_nfw+V_pt
    return V_total

### Analysis

# Define NFW halo properties
del_c_1, r_s_1, r_200_1, c_1, rho_s_1 = NFWprop(m_1,z_1,c_1)
del_c_2, r_s_2, r_200_2, c_2, rho_s_2 = NFWprop(m_2,z_2,c_2)

# Calculate the potential energy at observed time
PE_obs = PEnfwnfw(d_3d,m_1,rho_s_1,r_s_1,r_200_1,m_2,rho_s_2,r_s_2,r_200_2,N=del_mesh)

# Reduced mass
mu = m_1*m_2/(m_1+m_2)

# Total Energy
E = PE_obs+mu/2*v_3d_obs**2

# Calculate PE from d = 0 to r_200_1+r_200_2
del_TSM_mesh = (r_200_1+r_200_2)/(TSM_mesh-1)
d = numpy.arange(0.00001,(r_200_1+r_200_2)+del_TSM_mesh,del_TSM_mesh)
PE_array = numpy.zeros(TSM_mesh)
for j in numpy.arange(TSM_mesh):
    PE_array[j] = PEnfwnfw(d[j],m_1,rho_s_1,r_s_1,r_200_1,m_2,rho_s_2,r_s_2,r_200_2,N=del_mesh)

#*# Need to modify this original code to calculate the velocity various times
#*# during the merger (from t_col to t_obs)

# Calculate the 3D velocity at collision time
v_3d_col = numpy.sqrt(v_3d_obs**2+2/mu*(PE_obs-PE_col))

#*# Need to modify this original code to calculate TSM throughout the merger
#*# phase from (t_col to t_obs)

# Calculate TSM_0
if d_3d >= r_200_1+r_200_2:
    # then halos no longer overlap
    # calculate the time it takes to go from d=0 to r_200_1+r_200_2
    TSM_0a = numpy.sum(del_TSM_mesh/numpy.sqrt(2/mu*(E-PE_array))*kminMpc/sinGyr)
    # calculate th time it takes to go from d = r_200_1+r_200_2 tp d_3d
    TSM_0b = TSMptpt(m_1,m_2,r_200_1,r_200_2,d_3d,E)
    TSM_0 = TSM_0a+TSM_0b
else:
    # then d_3d < r_200_1+r_200_2, halos always overlap
    # calculate the time it takes to go from d=0 to d_3d            
    mask = d <= d_3d
    TSM_0 = numpy.sum(del_TSM_mesh/numpy.sqrt(2/mu*(E-PE_array[mask]))*kminMpc/sinGyr)
