#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:03:43 2020
@author: javier
"""

import numpy as np

def linmod_stoch(x0,tmax,params,q):
 t = range(0,tmax+1,1)
 xt = np.empty((tmax+1)); xt.fill(np.NaN)
 xt[0] = x0
 for j in range(tmax):
  xold = xt[j]
  xnew = params[0]*xold + params[1]*xold*np.abs(xold)     
  xt[j+1] = xnew + q*np.random.randn(1)
 return t,xt


def linmod_pf(x0_guess,t,tobs,y,r,M,q,params):
 # For the true time
 Nt = np.size(t)
 Ntobs = np.size(tobs)
 tstep_truth = t[1]-t[0]
 # For the analysis (we assimilate everytime we get observations)
 tstep_obs = tobs[1]-tobs[0]
 # The ratio
 o2t = int(tstep_obs/tstep_truth+0.5)
 # Precreate the arrays for background and analysis
 Xb = np.empty((Nt,M)); Xb.fill(np.nan)
 Xa = np.empty((Nt,M)); Xa.fill(np.nan)
 w = np.empty((Ntobs,M)); w.fill(np.nan)
 
 # For the original background ensemble
 back0 = 'fixed' # so we can generate repeatable experiments.
 desv = 1.0
 # Fixed initial conditions for our ensemble (created ad hoc)
 if back0=='fixed':
  Xb[0,:] = np.linspace(x0_guess-desv,x0_guess+desv,M)
 # Random initial conditions for our ensemble; let's perturb the
 # real x0 using errors with the magnitudes of R
 elif back0=='random':
  Xb[0,:] = x0_guess + desv*np.random.randn(M)
  
 # Since we don't have obs at t=0 the first analysis is the same as
 # background
 Xa[0,:] = Xb[0,:]
 w[0,:] = 1.0/M

 # The following cycle contains evolution and assimilation for all
 # the time steps
 for j in range(np.size(tobs)-1):
  # First we evolve the ensemble members
  # Evolve from analysis!
  xold = Xa[j*o2t,:] # [N,M]
  # Time goes forward
  xnew = evolvemembers(xold,tstep_truth,o2t,q,params) # needs [N,M] arrays,
  # The new background
  Xb[j*o2t+1:(j+1)*o2t+1,:] = xnew[1:,:] # [o2t,N,M]
  Xa[j*o2t+1:(j+1)*o2t+1,:] = xnew[1:,:] # [o2t,N,M]
  # The assimilation
  Xa_aux,waux = pf(Xb[(j+1)*o2t,:],y[j+1],r,M)
  Xa[(j+1)*o2t,:] = Xa_aux # introduce the auxiliary variable
  w[j+1,:] = waux                              
  print('t=', t[(j+1)*o2t])
 del j
 
 return Xb,Xa,w



##################################################################
def evolvemembers(xold,tstep_truth,o2t,Qsq,params):
 """Evolving the members.
 Inputs:  - xold, a [N,M] array of initial conditions for the
            M members and N variables
          - tstep_truth, the time step used in the nature run
          - o2t, frequency of observations in time steps
 Outputs: - xnew, a [o2t+1,N,M] array with the evolved members"""
 t_anal = o2t*tstep_truth
 M = np.size(xold)
 xnew = np.empty((o2t+1,M))
 xnew.fill(np.nan)
 for j in range(M):
  taux,xaux = linmod_stoch(xold[j],t_anal,params,Qsq) # [o2t+1,N]
  xnew[:,j] = xaux
 del j 
 return xnew

#
#%% The particle filter algorithm
def pf(Xb,y,r,M):
 w = np.empty((M)); w.fill(np.NaN)   
 for m in range(M):
  aux = (y - Xb[m])**2 / r**2
  w[m] = np.exp(-0.5*aux)
 del m   
 w = w/np.sum(w)
 xnew = Xb
 xnew = resample(Xb,w,M)
 return xnew,w 


#%% Resampling
def resample(Xb,w,M): 
 Xa = np.empty((M)); Xa.fill(np.NaN)
 
 ind_sort = np.argsort(w) 
 w_sort = np.sort(w)
 w_cumsort = np.cumsum(w_sort)
 
 ran_ori = 1.0/M * np.random.rand()
 lims = np.arange(ran_ori,1+1/M,1/M)
 
 for m in range(M):
  ind_aux = lims[m]<=w_cumsort
  candidates = ind_sort[ind_aux]
  actual = candidates[0] 
  Xa[m] = Xb[actual] 
 del m
 return Xa   



