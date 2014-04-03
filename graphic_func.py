import numpy as np
import jet_particle_functions as jpf
import matplotlib.pyplot as plt

N = jpf.N
DIM = jpf.DIM
SIGMA = jpf.SIGMA

def display_velocity( q , p ,mu ):
 W = 5*SIGMA
 res = 30
 N_nodes = res**DIM
 store = np.outer( np.linspace(-W,W , res), np.ones(res) )
 nodes = np.zeros( [N_nodes , DIM] )
 nodes[:,0] = np.reshape( store , N_nodes )
 nodes[:,1] = np.reshape( store.T , N_nodes )
 K,DK,D2K,D3K = jpf.derivatives_of_kernel(nodes , q)
 vel_field = np.einsum('ijab,jb->ia',K,p) - np.einsum('ijabc,jbc->ia',DK,mu)
 U = vel_field[:,0]
 V = vel_field[:,1]
 plt.figure()
 plt.quiver( nodes[:,0] , nodes[:,1] , U , V , scale=10 )
 plt.plot(q[:,0],q[:,1],'ro')
 for i in range(0,N):
     plt.arrow(q[i,0], q[i,1], p[i,0], p[i,1], head_width=0.1, head_length=0.2, lw = 4.0, fc='b', ec='b')
     plt.arrow(q[i,0], q[i,1], p[i,0], p[i,1], head_width=0.1, head_length=0.2, lw = 2.0, fc='w', ec='w')
 plt.axis('equal')
 plt.axis([- W, W,- W, W ])
 return plt.gcf()

def display_streamlines( q , p ,mu ):
 W = 5*SIGMA
 res = 30
 N_nodes = res**DIM
 store = np.outer( np.linspace(-W,W , res), np.ones(res) )
 nodes = np.zeros( [N_nodes , DIM] )
 nodes[:,0] = np.reshape( store , N_nodes )
 nodes[:,1] = np.reshape( store.T , N_nodes )
 K,DK,D2K,D3K = jpf.derivatives_of_kernel(nodes , q)
 vel_field = np.einsum('ijab,jb->ia',K,p) - np.einsum('ijabc,jbc->ia',DK,mu)
 U = vel_field[:,0]
 V = vel_field[:,1]
 U2 = U.reshape(res,res)
 V2 = V.reshape(res,res)
 Y,X = np.mgrid[-W:W:30j, -W:W:30j]
 speed = np.sqrt(U2*U2 + V2*V2)
 lw = 5*speed / speed.max()
 plt.figure()
 plt.streamplot( X , Y , U2.T , V2.T , color='k', linewidth=lw )
 plt.plot(q[:,0],q[:,1],'ro')
 for i in range(0,N):
     plt.arrow(q[i,0], q[i,1], 0.2*p[i,0], 0.2*p[i,1], head_width=0.05, head_length=0.1, fc='b', ec='b')
 plt.axis([- W, W,- W, W ])
 return plt.gcf()
 
def display_vorticity( q , p ,mu , quiver = None ):
 W = 5*SIGMA
 res = 100
 N_nodes = res**DIM
 store = np.outer( np.linspace(-W,W , res), np.ones(res) )
 nodes = np.zeros( [N_nodes , DIM] )
 nodes[:,0] = np.reshape( store , N_nodes )
 nodes[:,1] = np.reshape( store.T , N_nodes )
 K,DK,D2K,D3K = jpf.derivatives_of_kernel(nodes,q)
 J = np.zeros([2,2])
 J[0,1] = -1.
 J[1,0] = 1.
 vorticity = np.einsum('da,ijabd,jb->i',J,DK,p) - np.einsum('da,ijabcd,jbc->i',J,D2K,mu)
 Z = np.reshape(vorticity[:,], [res,res])
 X = np.reshape( nodes[:,0] , [res,res] )
 Y = np.reshape( nodes[:,1] , [res,res] )
 plt.figure()
 plt.contourf( X,Y, Z , 10, cmap=plt.cm.rainbow )
# plt.contour( X,Y, Z , 10, color='k' )
 plt.plot(q[:,0],q[:,1],'wo')
 for i in range(0,N):
     plt.arrow(q[i,0], q[i,1], 0.2*p[i,0], 0.2*p[i,1], head_width=0.05, head_length=0.1, fc='b', ec='b')
 if quiver=="on":
  res = 30
  N_nodes = res**DIM
  store = np.outer( np.linspace(-W,W , res), np.ones(res) )
  nodes = np.zeros( [N_nodes , DIM] )
  nodes[:,0] = np.reshape( store , N_nodes )
  nodes[:,1] = np.reshape( store.T , N_nodes )
  K,DK,D2K,D3K = jpf.derivatives_of_kernel(nodes , q)
  vel_field = np.einsum('ijab,jb->ia',K,p) - np.einsum('ijabc,jbc->ia',DK,mu)
  U = vel_field[:,0]
  V = vel_field[:,1]
  plt.quiver( nodes[:,0] , nodes[:,1] , U , V , scale = 20 )
 plt.axis([- W, W,- W, W ])
 plt.axis('equal')
 return plt.gcf()
