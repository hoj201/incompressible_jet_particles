import jet_particle_functions as jpf
import numpy as np

y_data = np.load('state_data.npy')
time_data = np.load('time_data.npy')

#print 'shape of y_data is ' + str( y_data.shape )
N_timestep = y_data.shape[0]
print 'generating png files'
for k in range(0,N_timestep):
	q,p,mu = jpf.state_to_weinstein_darboux( y_data[k] )
	fig = jpf.display_velocity_field(q,p,mu)
	fname = './movie_frames/frame_'+str(k)+'.png'
	fig.savefig( fname )
	fig.clf()
print 'done'
