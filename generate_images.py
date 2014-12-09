import jet_particle_functions as jpf
import graphic_func as gf
import matplotlib.pyplot as plt
import numpy as np
import sys
import multiprocessing as mp

y_data = np.load('./state_data.npy')
time_data = np.load('./time_data.npy')

def write_image(k):
	q,p,mu,q1 = jpf.state_to_weinstein_darboux( y_data[k] )
	f = gf.display_velocity(q,p,mu)
	time_s = str(time_data[k])
	plt.suptitle('t = '+ time_s[0:4] , fontsize=16 , x = 0.75 , y = 0.25 )
	fname = './movie_frames/frame_%03i.png' % k
	f.savefig( fname )
	plt.close(f)
	sys.stdout.write(' '+str(k))
	sys.stdout.flush()

N_timestep = y_data.shape[0]
sys.stdout.write('generating png files:')
sys.stdout.flush()
p = mp.Pool(8)
p.map(write_image,range(0,N_timestep))
sys.stdout.write(' done\n')
