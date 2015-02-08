#!/usr/bin/env python

import jet_particle_functions as jpf
import matplotlib.pyplot as plt
import numpy as np
import sys

N = jpf.N
DIM = jpf.DIM
SIGMA = jpf.SIGMA

if N != 1:
    exit(1)

y_data = np.load('./state_data.npy')
times  = np.load('./time_data.npy')

N_timestep = y_data.shape[0]

mu_comp = np.zeros([3,N_timestep]) # spin, stretch, shear

for i in range(N_timestep):
    q,p,mu,q1 = jpf.state_to_weinstein_darboux( y_data[i] )

    mu_comp[0][i] = (mu[0][1][0]-mu[0][0][1])/2
    mu_comp[1][i] = (mu[0][0][0]-mu[0][1][1])/2
    mu_comp[2][i] = (mu[0][0][1]+mu[0][1][0])/2

plt.figure()
plt.xlabel('t')
plt.plot(times,mu_comp[0],'b-',
         times,mu_comp[1],'r-',
         times,mu_comp[2],'g-')
plt.show()
