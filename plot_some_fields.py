import graphic_func as gf
import matplotlib.pyplot as plt
import numpy as np
from jet_particle_functions import N,SIGMA,DIM

if N == 1:
    q = np.zeros([N,DIM])
    p = np.zeros([N,DIM])
    p[0,0] = 1.
    mu = np.zeros([N,DIM,DIM])
    gf.display_velocity(q,p,mu)
    plt.show()
    plt.axis('equal')
    gf.display_vorticity(q,p,mu)
    plt.axis('equal')
    plt.xlabel('x')
    plt.show()
else:
    print 'N is not equal to 1, we opted to warn you of this rather than print any fields'
