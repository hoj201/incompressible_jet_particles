import graphic_func as gf
import matplotlib.pyplot as plt
import numpy as np
from jet_particle_functions import N,SIGMA,DIM
import jet_particle_functions as jpf

#coasting_zero_jet_data
#zero_jet_tackles_another
#dynamic_one_jet

#state = np.load('./movies/zero_jet_tackles_another.npy')
#time = np.load('./movies/zero_jet_tackles_another_time.npy')

#step = 100
spin = np.array([ [0. , -1.] , [1. , 0.]])
stretch = np.array([ [0. , 1.] , [1. , 0.]])
shear = np.array([ [0. , 0.] , [1. , 0.]])

q = np.zeros([N,DIM])
p = np.zeros([N,DIM])
mu = np.zeros([N,DIM,DIM])
mu[0] = spin
#p[0,0] = 1.
#q,p,mu = jpf.state_to_weinstein_darboux(state[step])
#print 'time = ' + str(time[step])

gf.display_velocity(q,p,mu)
plt.axis('equal')
plt.show()

gf.display_vorticity(q,p,mu,quiver="on")
plt.axis('equal')
plt.show()

#gf.display_streamlines(q,p,mu)
#plt.axis('equal')
#plt.show()
