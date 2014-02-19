import graphic_func as gf
import matplotlib.pyplot as plt
import numpy as np
from jet_particle_functions import N,SIGMA,DIM
import jet_particle_functions as jpf

state = np.load('./movies/dynamic_one_jet.npy')
time = np.load('./movies/dynamic_one_jet_time.npy')

step = 0
q,p,mu = jpf.state_to_weinstein_darboux(state[step])
print 'time = ' + str(time[step])

gf.display_velocity(q,p,mu)
plt.axis('equal')
plt.show()

gf.display_vorticity(q,p,mu)
plt.axis('equal')
plt.show()

gf.display_streamlines(q,p,mu)
plt.axis('equal')
plt.show()
