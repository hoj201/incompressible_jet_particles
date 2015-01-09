#!/usr/bin/env python

import jet_particle_functions as jpf
import matplotlib.pyplot as plt
import numpy as np
import sys

N = jpf.N
DIM = jpf.DIM
SIGMA = jpf.SIGMA

y_data = np.load('./state_data.npy')
times  = np.load('./time_data.npy')

N_timestep = y_data.shape[0]

momenta = np.zeros([N,3,N_timestep])

for i in range(N_timestep):
    q,p,mu,q1 = jpf.state_to_weinstein_darboux( y_data[i] )

    for j in range(N):
        tmp = jpf.lin_momentum(q,p,mu,[j])
        momenta[j][0][i] = tmp[0]
        momenta[j][1][i] = tmp[1]
        momenta[j][2][i] = jpf.ang_momentum(q,p,mu,[j])[0][1]

plt.figure()
plt.xlabel('t')
plt.plot(times,momenta[0][2],'b-',
         times,momenta[1][2],'r-',
         times,momenta[2][2],'g-')
#plt.axis([0,times[N_timestep-1],0,1])
plt.show()
