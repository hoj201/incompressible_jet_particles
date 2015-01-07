#!/usr/bin/env python

import numpy as np
import jet_particle_functions as jpf
#import matplotlib.pyplot as plt
#from display_vector_fields import display_velocity_field
from scipy.integrate import odeint

N = jpf.N # number of jetlets
DIM = jpf.DIM # dimension of space
SIGMA = jpf.SIGMA

spin = np.array( [ [0. , -1.] , [1. , 0.] ] )
stretch = np.array( [ [1. , 0.] , [0. , -1.] ] )
shear = np.array( [ [0. , 1.] , [0. , 0.] ] )

#q = np.random.rand(N,DIM) #initial positions of jetlets
q = np.zeros([N,DIM])
p = np.zeros([N,DIM])
mu = np.zeros([N,DIM,DIM])
q1 = np.zeros([N,DIM,DIM])
#p = np.random.rand(N,DIM) #initial 0-momentum of jetlets
#mu = np.random.rand(N,DIM,DIM)
#mu[0] = 0.5*spin
#mu[1] = -0.5*spin
#mu[0] = np.load('mu.npy')
#p = -q*np.ones([N,DIM])
#mu = np.random.randn(N,DIM,DIM)
#for i in range(0,N):
#    mu[i] = mu[i] - np.mean(np.diag(mu[i]))*np.eye(DIM)
for i in range(0,N):
    q1[i] = np.eye(DIM)

r0 = 3.0
p0 = -1.5
d = 0.33
omega = 0.0
q[0] = [-r0, 0 ]
q[1] = [ r0, 0 ]
#q[2] = [ -2, -4.5 ]
p[0] = [-p0, d ]
p[1] = [ p0,-d ]
#p[2][0] = 0.7
mu[0] =  omega*spin #+ 0.2*stretch
mu[1] = -omega*spin
#mu[2] = 0.2*spin

T = 60.0

#print 'testing various functions'
#print  jpf.test_functions(1)

print 'parameters:'
print 'r0 = %.2f' % r0
print 'p0 = %.2f' % p0
print 'd  = %.2f' % d
print 'omega = %.2f' % omega

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
step_max = 400
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
print 'dist = %.3e   p_0 = %.3e' % ( np.linalg.norm(q[1]-q[0]), np.linalg.norm(p[0]) )

print '  final J_R^1 momenta:'
Kf = np.zeros([N,DIM,DIM])
for i in range(0,N):
    Kf[i] = jpf.Jr1_momentum(q,p,mu,q1,particles=[i])
    print Kf[i]
#    print q1[i]
    print 'K[%2i] preserved? sum-abs = %3e' % (i,np.sum(np.abs(Ki[i]-Kf[i])))
