import jet_particle_functions as jpf
import graphic_func as gf
import matplotlib.pyplot as plt
import numpy as np

y_data = np.load('./state_data.npy')
time_data = np.load('./time_data.npy')

#y_data = np.load('state_data.npy')
#time_data = np.load('time_data.npy')
#print 'shape of y_data is ' + str( y_data.shape )
N_timestep = y_data.shape[0]
print 'generating png files'
for k in range(0,N_timestep):
	q,p,mu = jpf.state_to_weinstein_darboux( y_data[k] )
	f = gf.display_velocity(q,p,mu)
	time_s = str(time_data[k])
	plt.suptitle('t = '+ time_s[0:4] , fontsize=16 , x = 0.75 , y = 0.25 )
	fname = './movie_frames/frame_'+str(k)+'.png'
	f.savefig( fname )
	plt.close(f)
print 'done'
