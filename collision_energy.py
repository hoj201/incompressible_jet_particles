import numpy as np
import matplotlib.pyplot as plt

SIGMA = 1.0

def H(q,p):
    return (p**2)*(1+0.25*(1-(q**2)/(SIGMA**2))*np.exp(-(q**2)/(2*SIGMA**2)) )

res = 20
Q = np.outer( np.linspace(-1,1,res) , np.ones(res) ) 
P = np.outer( np.ones(res) , np.linspace(-1,1,res) ) 
E = H(Q,P)

plt.figure()
plt.contour( Q , P , E )
plt.show()
