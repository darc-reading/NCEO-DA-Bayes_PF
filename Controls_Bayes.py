# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 21:32:58 2017
@author: jamezcua
Last update: January 2020
Should work with Python 3
"""
import numpy as np
import matplotlib.pyplot as plt
from functions_Bayes import bayes_gm

# Computes the posterior distribution given a prior and a likelihood.
# Both are written as Gaussian mixtures

#%% ---------------------------------------
### 0.  The support of the distributions
x = np.arange(-7,7,0.05)


#%% ---------------------------------------
### 1. The components of the prior 
# The coefficients. Recall they must add to 1
alphas_prior = [.5,.5] 
# The means. 
mus_prior = [0,1]
# the standard deviations
sigmas_prior = [1,3]


#%%
### 2. The components of the likelihood
# the observation
y = 2.0
# the coefficients
alphas_like = [1.0] 
# The means. 
mus_like = [0]
# the standard deviations
sigmas_like = [1]


#%%
### 3. Computing the posterior
p_prior, p_like, p_post = bayes_gm(x,alphas_prior,mus_prior,sigmas_prior,\
                     y,alphas_like,mus_like,sigmas_like)


#%%
### 4. Plotting the results
plt.figure()
plt.plot(x,p_prior,'b',linewidth=2,label='prior')
plt.plot(x,p_like,'r',linewidth=2,label='likelihood')
plt.plot(x,p_post,'m',linewidth=2,label='posterior')
plt.xlabel('x',fontsize=14)
plt.ylabel('pdf',fontsize=14)
plt.title('Bayes theorem for Gaussian mixtures, h=I',fontsize=14)
plt.legend()





















