import numpy as np
import jet_particle_functions as jpf
#import matplotlib.pyplot as plt
#from display_vector_fields import display_velocity_field
from scipy.integrate import odeint

N = jpf.N
DIM = jpf.DIM
SIGMA = jpf.SIGMA

spin = np.array( [ [0. , -1.] , [1. , 0.] ] )
stretch = np.array( [ [1. , 0.] , [0. , -1.] ] )
shear = np.array( [ [0. , 1.] , [0. , 0.] ] )

q = np.random.rand(N,DIM)
#p = np.zeros([N,DIM])
p = np.random.rand(N,DIM)
mu = np.random.rand(N,DIM,DIM)
#mu[0] = 0.5*spin
#mu[1] = -0.5*spin
#mu[0] = np.load('mu.npy')
#p = -q*np.ones([N,DIM])
#mu = np.random.randn(N,DIM,DIM)
#for i in range(0,N):
#    mu[i] = mu[i] - np.mean(np.diag(mu[i]))*np.eye(DIM)

#print 'testing various functions'
#print  jpf.test_functions(1)

print 'initial energy is ' + str(jpf.Hamiltonian(q,p,mu))

state =  jpf.weinstein_darboux_to_state( q , p , mu )
step_max = 100
t_span = np.linspace( 0. , 4.0 , step_max )
y_span = odeint( jpf.ode_function , state , t_span , rtol=0.000001 )
np.save('state_data',y_span)
np.save('time_data',t_span)

q,p,mu = jpf.state_to_weinstein_darboux( y_span[step_max-1] )
print 'final energy is ' + str(jpf.Hamiltonian(q,p,mu))
