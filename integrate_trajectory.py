import numpy as np
import jet_particle_functions as jpf
#import matplotlib.pyplot as plt
#from display_vector_fields import display_velocity_field
from scipy.integrate import odeint

N = jpf.N
DIM = jpf.DIM
SIGMA = jpf.SIGMA

q = np.array( [ [-2.0,-0.5],[2.0,0.5] ] )
#p = np.zeros([N,DIM])
p = np.array( [ [1.0,0.0],[-1.0,0.0] ] )
#mu = np.zeros([N,DIM,DIM])
#mu[0,0,1] = -0.1
#mu[0,1,0] = 0.1
#mu[1,0,0] = 0.1
#mu[1,1,1] = -0.1
q = np.random.randn(N,DIM)
p = np.random.randn(N,DIM)
#p = -q*np.ones([N,DIM])
mu = np.random.randn(N,DIM,DIM)
for i in range(0,N):
    mu[i] = mu[i] - np.mean(np.diag(mu[i]))*np.eye(DIM)

s = jpf.weinstein_darboux_to_state( q , p , mu)
ds = jpf.ode_function( s , 0 )
dq,dp_coded,dmu = jpf.state_to_weinstein_darboux( ds ) 

print 'a test of the ode:'
print 'dp_coded =' + str(dp_coded)
h = 10e-7
Q = np.copy(q)
dp_estim = np.zeros([N,DIM])
for i in range(0,N):
    for a in range(0,DIM):
        Q[i,a] = q[i,a] + h
        dp_estim[i,a] = - ( jpf.Hamiltonian(Q,p,mu) - jpf.Hamiltonian(q,p,mu) ) / h 
        Q[i,a] = Q[i,a] - h
print 'dp_estim =' + str(dp_estim)
print 'dp_error =' + str(dp_estim - dp_coded)
print 'testing various functions'
print  jpf.test_functions(1)

print 'initial energy is ' + str(jpf.Hamiltonian(q,p,mu))

state =  jpf.weinstein_darboux_to_state( q , p , mu )
step_max = 100
t_span = np.linspace( 0. , 16.0 , step_max )
y_span = odeint( jpf.ode_function , state , t_span , rtol=0.000001 )
np.save('state_data',y_span)
np.save('time_data',t_span)

q,p,mu = jpf.state_to_weinstein_darboux( y_span[step_max-1] )
print 'final energy is ' + str(jpf.Hamiltonian(q,p,mu))
