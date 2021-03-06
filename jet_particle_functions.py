#from scipy.spatial.distance import pdist , squareform
import numpy as np

DIM = 2
N = 2
SIGMA = 1.0

# Basis for sl(2) and its dual
spin    = np.array( [ [0. ,-1.] , [1. , 0.] ] )
stretch = np.array( [ [1. , 0.] , [0. ,-1.] ] )
shear   = np.array( [ [0. , 1.] , [1. , 0.] ] )

import math

rho_large = 0.5 # Cut-off point below which to use Taylor series
diff_deg = 6 # Maximum degree of derivatives of F1, F2 (actually deg+1)

def scalar_F1( rho ):
    res = np.zeros( [diff_deg] )
    if rho > rho_large:
        g = np.exp(-rho)
        res[0] =  g - 0.5*(1. - g)/rho
        res[1] = -g + 0.5*(1. - g - rho*g )/ (rho**2)
        res[2] =  g -     (1. - g - rho*g - 0.5 * rho**2 * g )/ (rho**3)
        res[3] = -g + 3.0*(1. - g - rho*g - 0.5 * rho**2 * g - g*rho**3/6. )/ (rho**4)
        res[4] =  g -  12*(1. - g - rho*g - 0.5 * rho**2 * g - g*rho**3/6. - g*rho**4/24. )/ (rho**5)
        res[5] = -g +  60*(1. - g - rho*g - 0.5 * rho**2 * g - g*rho**3/6. - g*rho**4/24. - g*rho**5/120. )/ (rho**5)
    else:
        for d in range(diff_deg):
            for k in range(5):
                res[d] += (-1)**(k+d) * (1. - 1./(2*(k+d+1)) ) * rho**k / math.factorial(k)
    return tuple(res)

def scalar_F2( rho ):
    res = np.zeros( [diff_deg] )
    if rho > rho_large:
        g = np.exp(-rho)
        res[0] = 0.5*rho**(-2) * ( 1 - g - rho*g)
        res[1] = -   rho**(-3) * ( 1 - g - rho*g - 0.5* rho**2 * g)
        res[2] = 3.0*rho**(-4) * ( 1 - g - rho*g - 0.5* rho**2 * g - g*rho**3 / 6.)
        res[3] = -12*rho**(-5) * ( 1 - g - rho*g - 0.5* rho**2 * g - g*rho**3 / 6. - g*rho**4 / 24. )
        res[4] =  60*rho**(-6) * ( 1 - g - rho*g - 0.5* rho**2 * g - g*rho**3 / 6. - g*rho**4 / 24. - g*rho**5 / 120. )
        res[5] = 360*rho**(-7) * ( 1 - g - rho*g - 0.5* rho**2 * g - g*rho**3 / 6. - g*rho**4 / 24. - g*rho**5 / 120. - g*rho**6 / 720.)
    else:
        for d in range(diff_deg):
            for k in range(5):
                res[d] += 0.5*(-1)**(k+d) * 1./(k+d+2) * rho**k / math.factorial(k)
    return tuple(res)

def Hermite( k , x):
    #Calculate the 'statisticians' Hermite polynomials
    if k==0:
        return 1.
    elif k==1:
        return x
    elif k==2:
        return x**2 -1
    elif k==3:
        return x**3 - 3*x
    elif k==4:
        return x**4 - 6*x**2 + 3
    elif k==5:
        return x**5 - 10*x**3 + 15*x
    else:
        print 'error in Hermite function, unknown formula for k=' + str(k)

def derivatives_of_Gaussians( nodes , q ):
    #given x_i , x_j returns G(x_ij) for x_ij = x_i - x_j
    N_nodes = nodes.shape[0]
    dx = np.zeros([N_nodes,N,DIM])
    for i in range(0,N_nodes):
        for j in range(0,N):
            dx[i,j,:] = nodes[i,:] - q[j,:]

    rad_sq = np.einsum('ija,ija->ij',dx,dx)
    delta = np.eye(DIM)
    G = np.exp( -rad_sq / (2*SIGMA**2) )
    DG  = np.zeros([ N_nodes , N, DIM ]) #indices 0 and 1 are particle-indices
    D2G = np.zeros([ N_nodes , N, DIM , DIM ]) #indices 0 and 1 are particle-indices
    D3G = np.zeros([ N_nodes , N, DIM , DIM , DIM ]) #indices 0 and 1 are particle-indices

    DG = np.einsum('ija,ij->ija',-dx/(SIGMA**2) , G )
    D2G = (1./SIGMA**2)*(np.einsum('ab,ij->ijab',-delta , G ) - np.einsum('ija,ijb->ijab',dx , DG ) )
    D3G = (1./SIGMA**2)*(np.einsum('ab,ijc->ijabc',-delta,DG) - np.einsum('ac,ijb->ijabc',delta,DG) - np.einsum('ija,ijcb->ijabc',dx,D2G) )
    return G , DG, D2G, D3G

def derivatives_of_kernel( nodes , q ):
    #given x_i and x_j the K = Kernel( x_ij) and derivatives with x_ij = x_i - x_j.
    N_nodes = nodes.shape[0]
    x = np.zeros([DIM])
    #The code is written such that we evaluate at the nodes, and entry (i,j) is the contribution at node i due to particle j.
    delta = np.identity( DIM )
    K   = np.zeros( [N_nodes, N , DIM , DIM ] )
    DK  = np.zeros( [N_nodes, N , DIM , DIM, DIM ] )
    D2K = np.zeros( [N_nodes, N , DIM , DIM, DIM, DIM ] )
    D3K = np.zeros( [N_nodes, N , DIM , DIM, DIM, DIM, DIM ] )
    for i in range(0,N_nodes):
        for j in range(0,N):
            x = nodes[i] - q[j]
            rho = np.dot(x,x)/2
            F1,DF1,D2F1,D3F1,D4F1,D5F1 = scalar_F1( rho )
            F2,DF2,D2F2,D3F2,D4F1,D5F1 = scalar_F2( rho )
            K[i,j] = F1*delta + F2*np.einsum('a,b->ab',x,x)
            DK[i,j]  =  DF1* np.einsum('c,ab->abc',x,delta) \
                     +  DF2* np.einsum('a,b,c->abc',x,x,x) \
                     +   F2*(np.einsum('ac,b->abc',delta,x) \
                            +np.einsum('a,bc->abc',x,delta))
            D2K[i,j] = D2F1* np.einsum('d,c,ab->abcd',x,x,delta) \
                     +  DF1* np.einsum('cd,ab->abcd',delta,delta) \
                     + D2F2* np.einsum('d,c,a,b->abcd',x,x,x,x) \
                     +  DF2*(np.einsum('cd,a,b->abcd',delta,x,x) \
                            +np.einsum('c,ad,b->abcd',x,delta,x) \
                            +np.einsum('c,a,bd->abcd',x,x,delta) \
                            +np.einsum('d,ac,b->abcd',x,delta,x) \
                            +np.einsum('d,a,bc->abcd',x,x,delta)) \
                     +   F2*(np.einsum('ac,bd->abcd',delta,delta) \
                            +np.einsum('ad,bc->abcd',delta,delta))
            D3K[i,j] = D3F1* np.einsum('e,d,c,ab->abcde',x,x,x,delta) \
                     + D2F1*(np.einsum('c,de,ab->abcde',x,delta,delta) \
                            +np.einsum('d,ec,ab->abcde',x,delta,delta) \
                            +np.einsum('e,cd,ab->abcde',x,delta,delta)) \
                     + D3F2* np.einsum('e,d,c,a,b->abcde',x,x,x,x,x) \
                     + D2F2*(np.einsum('ed,c,a,b->abcde',delta,x,x,x) \
                            +np.einsum('ec,d,a,b->abcde',delta,x,x,x) \
                            +np.einsum('cd,e,a,b->abcde',delta,x,x,x) \
                            +np.einsum('d,c,ae,b->abcde',x,x,delta,x) \
                            +np.einsum('e,c,ad,b->abcde',x,x,delta,x) \
                            +np.einsum('e,d,ac,b->abcde',x,x,delta,x) \
                            +np.einsum('d,c,a,be->abcde',x,x,x,delta) \
                            +np.einsum('c,e,a,bd->abcde',x,x,x,delta) \
                            +np.einsum('e,d,a,bc->abcde',x,x,x,delta)) \
                     +  DF2*(np.einsum('b,cd,ea->abcde',x,delta,delta) \
                            +np.einsum('b,ce,da->abcde',x,delta,delta) \
                            +np.einsum('b,de,ac->abcde',x,delta,delta) \
                            +np.einsum('a,dc,eb->abcde',x,delta,delta) \
                            +np.einsum('a,ce,db->abcde',x,delta,delta) \
                            +np.einsum('a,ed,bc->abcde',x,delta,delta) \
                            +np.einsum('c,ad,be->abcde',x,delta,delta) \
                            +np.einsum('c,ea,db->abcde',x,delta,delta) \
                            +np.einsum('d,ac,be->abcde',x,delta,delta) \
                            +np.einsum('d,ae,cb->abcde',x,delta,delta) \
                            +np.einsum('e,ac,bd->abcde',x,delta,delta) \
                            +np.einsum('e,ad,bc->abcde',x,delta,delta))

    #EXAMPLE OF INDEX CONVENTION 'ijabc' refers to the c^th derivative of the ab^th entry of K(q_i - q_j)
    return K, DK, D2K, D3K

def Hamiltonian( q , p , mu ):
    #return the Hamiltonian.  Serves as a safety to check our equations of motion are correct.
    K,DK,D2K,D3K = derivatives_of_kernel(q,q)
    term_00 = 0.5*np.einsum('ia,ijab,jb',p,K,p)
    term_01 = - np.einsum('ia,ijabc,jbc',p,DK,mu)
    term_11 = - 0.5*np.einsum('iac,ijabcd,jbd',mu,D2K,mu)
    return term_00 + term_01 + term_11

def lin_momentum( q , p , mu, particles = range(N) ):
    # Returns linear (spatial) momentum of set of particles, defaults
    # to all particles, i.e. total momentum.
    res = np.zeros( [DIM] )
    for i in particles: res += p[i]
    return res

def ang_momentum( q , p , mu, particles = range(N) ):
    # Returns angular (spatial) momentum of set of particles, defaults
    # to all particles, i.e. total angular momentum.
    res = np.zeros( [DIM,DIM] )
    for i in particles:
        for a in range(DIM):
            for b in range(a):
                tmp = p[i][a]*q[i][b] - p[i][b]*q[i][a] + mu[i][a][b] - mu[i][b][a]
                res[a][b] += tmp
                res[b][a] -= tmp
    return res

def Jr1_momentum( q , p , mu, q1 , particles = range(N) ):
    # Returns momentum associated to right symmetry of iso(z) at 1-jet
    # level of a set of particles, defaults to all particles.
    res = np.zeros( [DIM,DIM] )
    for i in particles:
        a = q1[i]
        ainv = np.linalg.inv(a)
        res += -np.einsum('ac,ab,db->cd',a,mu[i],ainv)
    return res

def ode_function( state , t ):
    q , p , mu , q1 = state_to_weinstein_darboux( state )
    K,DK,D2K,D3K = derivatives_of_kernel( q , q )
    dq = np.einsum('ijab,jb->ia',K,p) - np.einsum('ijabc,jbc->ia',DK,mu)
    xi = np.einsum('ijacb,jc->iab',DK,p) - np.einsum('ijacbd,jcd->iab',D2K,mu)
    chi = np.einsum('ijadbc,jd->iabc',D2K,p) - np.einsum('ijaebcd,jed->iabc',D3K,mu)
    dp = - np.einsum('ib,jc,ijbca->ia',p,p,DK) \
        + np.einsum('id,jbc,ijdbca->ia',p,mu,D2K) \
        - np.einsum('jd,ibc,ijdbca->ia',p,mu,D2K) \
        + np.einsum('icb,jed,ijceabd->ia',mu,mu,D3K)
    dmu = np.einsum('iac,ibc->iab',mu,xi) - np.einsum('icb,ica->iab',mu,xi)
    dq1 = np.einsum('iab,ibc->iac',xi,q1)
    dstate = weinstein_darboux_to_state( dq , dp , dmu , dq1 )
    return dstate

def state_to_weinstein_darboux( state ):
    q = np.reshape( state[0:(N*DIM)] , [N,DIM] )
    p = np.reshape( state[(N*DIM):(2*N*DIM)] , [N,DIM] )
    mu = np.reshape( state[(2*N*DIM):(2*N*DIM + N*DIM*DIM)] , [N,DIM,DIM] )
    q1 = np.reshape( state[(2*N*DIM + N*DIM*DIM):(2*N*DIM + 2*N*DIM*DIM)] , [N,DIM,DIM] )
    return q , p , mu , q1

def weinstein_darboux_to_state( q , p , mu , q1 ):
    state = np.zeros( 2*N*DIM + 2*N*DIM*DIM )
    state[0:(N*DIM)] = np.reshape( q , N*DIM )
    state[(N*DIM):(2*N*DIM)] = np.reshape( p , N*DIM )
    state[(2*N*DIM):(2*N*DIM+N*DIM*DIM)] = np.reshape( mu , N*DIM*DIM)
    state[(2*N*DIM+N*DIM*DIM):(2*N*DIM+2*N*DIM*DIM)] = np.reshape( q1 , N*DIM*DIM)
    return state


def test_functions( trials ):
    #checks that each function does what it is supposed to
    
    #testing derivatives of Gaussians
    h = 10e-7
    q = SIGMA*np.random.randn(N,DIM)
    p = SIGMA*np.random.randn(N,DIM)
    mu = np.random.randn(N,DIM,DIM)
    G,DG,D2G,D3G = derivatives_of_Gaussians(q,q)
    q_a = np.copy(q)
    q_b = np.copy(q)
    q_c = np.copy(q)

    for i in range(0,N):
        for a in range(0,DIM):
            error_max = 0.
            q_a[i,a] = q[i,a]+h
            G_a , DG_a , D2G_a , D3G_a = derivatives_of_Gaussians(q_a, q) 
            for j in range(0,N):
                error = (G_a[i,j] - G[i,j])/h - DG[i,j,a]
                error_max = np.maximum( np.absolute( error ) , error_max )
            print 'max error for DG was ' + str( error_max )
            error_max = 0.
            for b in range(0,DIM):
                q_b[i,b] = q[i,b] + h
                G_b , DG_b , D2G_b , D3G_b = derivatives_of_Gaussians( q_b , q ) 
                for j in range(0,N):
                    error = (DG_b[i,j,a] - DG[i,j,a])/h - D2G[i,j,a,b]
                    error_max = np.maximum( np.absolute( error ) , error_max )
                print 'max error for D2G was ' + str( error_max )
                error_max = 0.
                for c in range(0,DIM):
                    q_c[i,c] = q_c[i,c] + h
                    G_c , DG_c , D2G_c , D3G_c = derivatives_of_Gaussians( q_c , q ) 
                    for j in range(0,N):
                        error = (D2G_c[i,j,a,b] - D2G[i,j,a,b])/h - D3G[i,j,a,b,c]
                        error_max = np.maximum( np.absolute(error) , error_max )
                    print 'max error for D3G was ' + str( error_max )
                    q_c[i,c] = q_c[i,c] - h
                q_b[i,b] = q_b[i,b] - h
            q_a[i,a] = q_a[i,a] - h
    

    K,DK,D2K,D3K = derivatives_of_kernel(q,q)
    delta = np.identity(DIM)
    error_max = 0.
    for i in range(0,N):
        for j in range(0,N):
            x = q[i,:] - q[j,:]
            r_sq = np.inner( x , x )
            for a in range(0,DIM):
                for b in range(0,DIM):
                    G = np.exp( -r_sq / (2.*SIGMA**2) )
                    K_ij_ab = ( (x[a]*x[b])/(SIGMA**2) + (1. - r_sq/(SIGMA**2) )*delta[a,b] )*G
                    error = K_ij_ab - K[i,j,a,b]
                    error_max = np.maximum( np.absolute(error) , error_max )

    print 'error_max for K = ' + str( error_max )
    if (error_max > 100*h):
        print 'WARNING:  COMPUTATION OF K APPEARS TO BE INACCURATE'

    error_max = 0.
    for i in range(0,N):
        for a in range(0,DIM):
            q_a[i,a] = q[i,a] + h
            K_a,DK_a,D2K_a,D3K_a = derivatives_of_kernel(q_a,q)
            for j in range(0,N):
                der = ( K_a[i,j,:,:] - K[i,j,:,:] ) / h
                error = np.linalg.norm(  der - DK[i,j,:,:,a] )
                error_max = np.maximum(error, error_max)
            q_a[i,a] = q[i,a]
    print 'error_max for DK = ' + str( error_max )
    if (error_max > 100*h):
        print 'WARNING:  COMPUTATION OF DK APPEARS TO BE INACCURATE'

    error_max = 0.
    q_b = np.copy(q)
    for i in range(0,N):
        for a in range(0,DIM):
            for b in range(0,DIM):
                q_b[i,b] = q[i,b] + h
                K_b,DK_b,D2K_b,D3K_b = derivatives_of_kernel(q_b,q)
                for j in range(0,N):
                    der = (DK_b[i,j,:,:,a] - DK[i,j,:,:,a] )/h
                    error = np.linalg.norm( der - D2K[i,j,:,:,a,b] )
                    error_max = np.maximum( error, error_max )
                q_b[i,b] = q[i,b]

    print 'error_max for D2K = ' + str( error_max )
    if (error_max > 100*h):
        print 'WARNING:  COMPUTATION OF D2K APPEARS TO BE INACCURATE'

    error_max = 0.
    q_c = np.copy(q)
    for i in range(0,N):
        for a in range(0,DIM):
            for b in range(0,DIM):
                for c in range(0,DIM):
                    q_c[i,c] = q[i,c] + h
                    K_c,DK_c,D2K_c,D3K_c = derivatives_of_kernel(q_c,q)
                    for j in range(0,N):
                        der = (D2K_c[i,j,:,:,a,b] - D2K[i,j,:,:,a,b] )/h
                        error = np.linalg.norm( der - D3K[i,j,:,:,a,b,c] )
                        error_max = np.maximum( error, error_max )
                    q_c[i,c] = q[i,c]

    print 'error_max for D3K = ' + str( error_max )

    if (error_max > 100*h):
        print 'WARNING:  COMPUTATION OF D3K APPEARS TO BE INACCURATE'

    print 'TESTING SYMMETRIES'
    print 'Is K symmetric with respect to ij?'
    error_max = 0
    for i in range(0,N):
        for j in range(0,N):
            error = np.linalg.norm( K[i,j,:,:] - K[j,i,:,:] )
            error_max = np.maximum( error, error_max )
    print 'max for K_ij - K_ji = ' + str( error_max )

    print 'Is DK anti-symmetric with respect to ij?'
    error_max = 0
    for i in range(0,N):
        for j in range(0,N):
            for a in range(0,DIM):
                error = np.linalg.norm( DK[i,j,:,:,a] + DK[j,i,:,:,a] )
                error_max = np.maximum( error, error_max )
    print 'max for DK_ij + DK_ji = ' + str( error_max )



    s = weinstein_darboux_to_state( q , p , mu ) # FIXME: add q1
    ds = ode_function( s , 0 )
    dq,dp_coded,dmu = state_to_weinstein_darboux( ds ) 

    print 'a test of the ode:'
    print 'dp_coded =' + str(dp_coded)

    Q = np.copy(q)
    dp_estim = np.zeros([N,DIM])
    for i in range(0,N):
        for a in range(0,DIM):
            Q[i,a] = q[i,a] + h
            dp_estim[i,a] = - ( Hamiltonian(Q,p,mu) - Hamiltonian(q,p,mu) ) / h 
            Q[i,a] = Q[i,a] - h
    print 'dp_estim =' + str(dp_estim)
    print 'dp_error =' + str(dp_estim - dp_coded)

    return 'what do you think?'
