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
#p = np.random.rand(N,DIM) #initial 0-momentum of jetlets
#mu = np.random.rand(N,DIM,DIM)
#mu[0] = 0.5*spin
#mu[1] = -0.5*spin
#mu[0] = np.load('mu.npy')
#p = -q*np.ones([N,DIM])
#mu = np.random.randn(N,DIM,DIM)
#for i in range(0,N):
#    mu[i] = mu[i] - np.mean(np.diag(mu[i]))*np.eye(DIM)

r0 = 3.0
p0 = -1.5
d = 1.2*SIGMA
J = d*p0
omega = 0.1
q[0] = [-r0,-d/2]
q[1] = [ r0, d/2]
#q[2] = [ -2, -3.5 ]
p[0][0] = -p0
p[1][0] =  p0
#p[2][0] =  0.4
mu[0] =  omega*spin
mu[1] = -omega*spin
#mu[2] = 0.2*spin

T = 60.0

#print 'testing various functions'
#print  jpf.test_functions(1)

def momentum(q,p,mu):
	res = [ 0,0 ]
	for i in range(N): res += p[i]
	return res


E0 = jpf.Hamiltonian(q,p,mu)
p0 = momentum(q,p,mu)
print 'initial energy is ' + str(E0)
print 'initial momentum: ' + str(p0[0]) + ',' + str(p0[1])

state =  jpf.weinstein_darboux_to_state( q , p , mu )
step_max = 200
t_span = np.linspace( 0. , T , step_max )
y_span = odeint( jpf.ode_function , state , t_span , rtol=0.000001 )
np.save('state_data',y_span)
np.save('time_data',t_span)

q,p,mu = jpf.state_to_weinstein_darboux( y_span[step_max-1] )
Ef = jpf.Hamiltonian(q,p,mu)
pf = momentum(q,p,mu)
print '  final energy is ' + str(Ef) + '  diff = ' + str(Ef-E0)
print '  final momentum: ' + str(p0[0]) + ',' + str(p0[1])
