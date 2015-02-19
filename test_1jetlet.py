#!/usr/bin/env python

import numpy as np
import jet_particle_functions as jpf
#import matplotlib.pyplot as plt
#from display_vector_fields import display_velocity_field
from scipy.integrate import odeint

N = jpf.N # number of jetlets
DIM = jpf.DIM # dimension of space
SIGMA = jpf.SIGMA

if N != 1:
    print 'This simulation requires N=1.'
    exit(1)

# Initialize 1-jetlet
q = np.zeros([N,DIM])
p = np.zeros([N,DIM])
mu = np.zeros([N,DIM,DIM])
q1 = np.zeros([N,DIM,DIM])
for i in range(0,N):
    q1[i] = np.eye(DIM)

# Initial 1-momentum of jetlets
mu[0] =  -0.84*spin + 0.89*stretch

T = 15.0

Ei = jpf.Hamiltonian(q,p,mu)
pi = jpf.lin_momentum(q,p,mu)
Li = jpf.ang_momentum(q,p,mu)
print 'initial energy: %.3f' % Ei
print 'initial momentum: %.3f,%.3f  %.3f' % ( pi[0], pi[1], Li[0][1] )

print 'initial J_R^1 momenta:'
Ki = np.zeros([N,DIM,DIM])
for i in range(0,N):
    Ki[i] = jpf.Jr1_momentum(q,p,mu,q1,particles=[i])
    print Ki[i]

state =  jpf.weinstein_darboux_to_state( q , p , mu , q1 )
step_max = 200
t_span = np.linspace( 0. , T , step_max )
y_span = odeint( jpf.ode_function , state , t_span , rtol=0.0000001 )
np.save('state_data',y_span)
np.save('time_data',t_span)

q,p,mu,q1 = jpf.state_to_weinstein_darboux( y_span[step_max-1] )

Ef = jpf.Hamiltonian(q,p,mu)
pf = jpf.lin_momentum(q,p,mu)
Lf = jpf.ang_momentum(q,p,mu)
print '  final energy: %.3f  diff = %.3e' % ( Ef, Ef-Ei )
print '  final momentum: %.3f,%.3f  %.3f  diff = %.3e %.3e' % \
    ( pf[0], pf[1], Lf[0][1], np.linalg.norm(pf-pi), np.fabs(Lf[0][1]-Li[0][1]) )
print '  final position: %.2f,%.2f' % ( q[0][0], q[0][1] )

print '  final J_R^1 momenta:'
Kf = np.zeros([N,DIM,DIM])
for i in range(0,N):
    Kf[i] = jpf.Jr1_momentum(q,p,mu,q1,particles=[i])
    print Kf[i]
#    print q1[i]
    print 'K[%2i] preserved? sum-abs = %3e  |Q| = %3e' % \
        (i,np.sum(np.abs(Ki[i]-Kf[i])),np.max(np.abs(q1[i])))
