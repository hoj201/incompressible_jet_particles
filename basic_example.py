import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def update_line(num , data , line):
    global x_nodes, y_nodes
    line.set_data(data[:,num])
    U = data[0,num]*np.ones(100)
    V = data[1,num]*np.ones(100)
    plt.quiver(x_nodes,y_nodes,U,V)
    return line,

fig1 = plt.figure()

data = np.zeros([2, 25])
data[0,:] = np.cos( np.linspace(0,7,25))
data[1,:] = np.sin( np.linspace(0,7,25))
l, = plt.plot([], [], 'ro')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel('x')
plt.title('test')

store = np.outer( np.linspace(-1,1,10) , np.ones(10) )
x_nodes = np.reshape( store , 100 )
y_nodes = np.reshape( store.T , 100 )
U = np.ones(100)
V = np.zeros(100)
plt.quiver(x_nodes,y_nodes,U,V)

line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
    interval=50, blit=True)
#line_ani.save('lines.mp4')

plt.show()
