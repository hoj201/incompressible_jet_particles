import numpy as np
import jet_particle_functions as kf
import matplotlib.pyplot as plt

N = jpf.N
DIM = jpf.DIM
SIGMA = jpf.SIGMA

def display_velocity_field( q , p ,mu ):
 W = 5*SIGMA
 res = 30
 N_nodes = res**DIM
 store = np.outer( np.linspace(-W,W , res), np.ones(res) )
 nodes = np.zeros( [N_nodes , DIM] )
 nodes[:,0] = np.reshape( store , N_nodes )
 nodes[:,1] = np.reshape( store.T , N_nodes )
 K,DK,D2K,D3K = kf.derivatives_of_kernel(q , nodes)
 vel_field = np.einsum('ijab,jb->ia',K,p) - np.einsum('ijabc,jbc->ia',DK,mu)
 U = vel_field[:,0]
 V = vel_field[:,1]

 plt.figure()
 plt.quiver( nodes[:,0] , nodes[:,1] , U , V )
 plt.plot(q[:,0],q[:,1],'ro')
 for i in range(0,N):
     plt.arrow(q[i,0], q[i,1], 0.2*p[i,0], 0.2*p[i,1], head_width=0.05, head_length=0.1, fc='b', ec='b')
 plt.axis([- W, W,- W, W ])
 return plt.gcf()
