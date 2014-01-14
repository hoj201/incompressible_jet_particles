import jet_particle_functions as jpf
import matplotlib.pyplot as plt
import numpy as np

def display_velocity_field( q , p ,mu ):
	W = 5*jpf.SIGMA
	res = 30
	N_nodes = res**jpf.DIM
	store = np.outer( np.linspace(-W,W , res), np.ones(res) )
	nodes = np.zeros( [N_nodes , jpf.DIM] )
	nodes[:,0] = np.reshape( store , N_nodes )
	nodes[:,1] = np.reshape( store.T , N_nodes )
	K,DK,D2K,D3K = jpf.derivatives_of_kernel( nodes , q )
	vel_field = np.einsum('ijab,jb->ia',K,p) - np.einsum('ijabc,jbc->ia',DK,mu)
	U = vel_field[:,0]
	V = vel_field[:,1]
	f = plt.figure(1)
	plt.quiver( nodes[:,0] , nodes[:,1] , U , V , color='0.50' )
	plt.plot(q[:,0],q[:,1],'ro')
	for i in range(0,len(q)):
		plt.arrow(q[i,0], q[i,1], 0.2*p[i,0], 0.2*p[i,1],\
				  head_width=0.2, head_length=0.2,\
				  fc='b', ec='b')
	plt.axis([- W, W,- W, W ])
	return f

y_data = np.load('state_data.npy')
time_data = np.load('time_data.npy')
#print 'shape of y_data is ' + str( y_data.shape )
N_timestep = y_data.shape[0]
print 'generating png files'
for k in range(0,N_timestep):
	q,p,mu = jpf.state_to_weinstein_darboux( y_data[k] )
	f = display_velocity_field(q,p,mu)
	time_s = str(time_data[k])
	plt.suptitle('t = '+ time_s[0:4] , fontsize=16 , x = 0.75 , y = 0.25 )
	fname = './movie_frames/frame_'+str(k)+'.png'
	f.savefig( fname )
	plt.close(f)
print 'done'
