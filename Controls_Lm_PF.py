# Last review, Jan 2020. Should work with python 3

import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt
from functions_Lm_Pf import linmod_stoch, linmod_pf

# this code illustrates the application of a SIR particle filter in a 0D nonlinear model with model error

#%% 1. Nature run
x0 = 0.1 # initial condition
tmax = 30 # maximum time
params = [0.96,.05] # linear and non-linear coefficient
q = .1 # std of the model error
t, xt = linmod_stoch(x0,tmax,params,q)

fsize=14
plt.figure()
plt.plot(t,xt,'-k.')
plt.xlabel('time',fontsize=fsize-1)
plt.ylabel('x_t',fontsize=fsize-1)
plt.title('$x_t=$'+str(params[0])+'$x_{t-1}+$'+str(params[1])+'$x_{t-1}|x_{t-1}|+$'+'$v_t$',fontsize=fsize)
plt.tick_params(labelsize=fsize-2) 


#%% 2. Observations every period_obs
period_obs = 5 # observation period
tobs = range(0,tmax+period_obs,period_obs)
Ntobs = np.size(tobs); 
r = 0.05 # observational error standard deviation
y = np.empty((Ntobs)); y.fill(np.NaN)
y[1:] = xt[tobs[1:]] + r*np.random.randn(Ntobs-1) 


fsize=14
plt.figure()
plt.plot(t,xt,'-k.',label='truth')
plt.scatter(tobs,y,s=50,c='r',label='observations')
plt.xlabel('time',fontsize=fsize-1)
plt.ylabel('x_t',fontsize=fsize-1)
plt.title('$x_t=$'+str(params[0])+'$x_{t-1}+$'+str(params[1])+'$x_{t-1}|x_{t-1}|+$'+'$v_t$',fontsize=fsize)
plt.tick_params(labelsize=fsize-2) 
plt.legend()


    
##############################################################################
### 3. Data assimilation using PFs     
Ne = 10; # number of particles
# A guess to start from in our assimilation experiments
x0guess = 0.1
Xb,Xa,w = linmod_pf(x0guess,t,tobs,y,r,Ne,q,params)

indM = range(1,Ne+1,1)
Nwin = int(np.size(xt)/period_obs)


fsize=14
plt.figure()
plt.plot(t,xt,'-k.',label='truth')
for j in range(Nwin):
 plt.plot(t[j*period_obs+1:(j+1)*period_obs+1],Xb[j*period_obs+1:(j+1)*period_obs+1,:],color='b',linestyle='--',label='bgd')
del j    
plt.plot(tobs,y,color='r',marker='o',linestyle='',label='obs')
plt.plot(tobs,Xb[tobs,:],color='b',marker='.',linestyle='')
plt.plot(t,Xa,'-m.',label='ana')
plt.tick_params(labelsize=fsize) 
plt.xlabel('time')
plt.ylabel('x_t')
plt.title('SIR PF in a 1D model, Ne='+str(Ne))
plt.legend()


Nana = np.size(tobs)
fsize = 12
plt.figure()
for j in range(Nana):
 plt.subplot(2,5,1+j)   
 plt.scatter(indM,w[j,:],c='m',s=40)
 plt.xlabel('particle',fontsize=fsize-1)
 plt.ylabel('weight',fontsize=fsize-1)
 plt.title('t='+str(tobs[j]),fontsize=fsize)
 plt.tick_params(labelsize=fsize-2)
 plt.ylim([-0.05,1.0]) 
del j
plt.subplots_adjust(top=0.955,bottom=0.08,left=0.11,right=0.9,hspace=0.265,wspace=0.345)













