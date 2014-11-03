#from scipy.spatial.distance import pdist , squareform
import numpy as np
from scipy.integrate import odeint
import kernels.pyGaussian as gaussian

N = 2
DIM = 2
SIGMA = 1.0

def getN():
    return N

def getDIM():
    return DIM

def get_dim_state():
    return 2*N*(DIM + DIM**2 + DIM**3)

def get_dim_adj_state():
    return 2*dim_state

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
    DG = np.zeros([ N_nodes , N, DIM ]) #indices 0 and 1 are particle-indices
    D2G = np.zeros([ N_nodes , N, DIM , DIM ]) #indices 0 and 1 are particle-indices
    D3G = np.zeros([ N_nodes , N, DIM , DIM , DIM ]) #indices 0 and 1 are particle-indices

    DG = np.einsum('ija,ij->ija',-dx/(SIGMA**2) , G )
    D2G = (1./SIGMA**2)*(np.einsum('ab,ij->ijab',-delta , G ) - np.einsum('ija,ijb->ijab',dx , DG ) )
    D3G = (1./SIGMA**2)*( np.einsum('ab,ijc->ijabc',-delta,DG) - np.einsum('ac,ijb->ijabc',delta,DG) - np.einsum('ija,ijcb->ijabc',dx,D2G) )

    return G , DG, D2G, D3G

    
def derivatives_of_kernel( nodes , q ):
    #given x_i and x_j the K = Kernel( x_ij) and derivatives with x_ij = x_i - x_j.
    #The code is written such that we evaluate at the nodes, and entry (i,j) is the contribution at node i due to particle j.
    delta = np.identity( DIM )
    G,DG,D2G,D3G,D4G,D5G,D6G = gaussian.derivatives_of_Gaussians( nodes , q )
    K = np.einsum('ij,ab->ijab',G,delta)
    DK = np.einsum('ijc,ab->ijabc',DG,delta)
    D2K = np.einsum('ijcd,ab->ijabcd',D2G,delta)
    D3K = np.einsum('ijcde,ab->ijabcde',D3G,delta)
    D4K = np.einsum('ijcdef,ab->ijabcdef',D4G,delta)
    D5K = np.einsum('ijcdefg,ab->ijabcdefg',D5G,delta)
    D6K = np.einsum('ijcdefgh,ab->ijabcdefgh',D6G,delta)
    #EXAMPLE OF INDEX CONVENTION 'ijabc' refers to the c^th derivative of the ab^th entry of K(q_i - q_j)
    return K, DK, D2K, D3K , D4K , D5K , D6K

def Hamiltonian( q , p , mu_1 , mu_2 ):
    #returns the Hamiltonian.  Serves as a safety to check our equations of motion are correct.
    K,DK,D2K,D3K,D4K,D5K,D6G = derivatives_of_kernel(q,q)
    term_00 = 0.5*np.einsum('ia,ijab,jb',p,K,p)
    term_01 = - np.einsum('ia,ijabc,jbc',p,DK,mu_1)
    term_11 = -0.5*np.einsum('iad,ijabcd,jbc',mu_1,D2K,mu_1)
    term_02 = np.einsum('ia,jbcd,ijabcd',p,mu_2,D2K)
    term_12 = np.einsum('iae,jbcd,ijabecd',mu_1,mu_2,D3K)
    term_22 = 0.5*np.einsum('iaef,jbcd,ijabcdef',mu_2,mu_2,D4K)
    return term_00 + term_01 + term_11 + term_02 + term_12 + term_22
    
def grad_Hamiltonian( q , p , mu_1 , mu_2 ):
    # the (p,mu_1,mu_2) gradient of the Hamiltonian
    K,DK,D2K,D3K,D4K,D5K,D6G = derivatives_of_kernel(q,q)
    g_p = np.einsum('ijab,jb->ia',K,p) # term_00
    g_p = g_p - np.einsum('ijabc,jbc->ia',DK,mu_1) # term_01
    g_mu_1 = - np.einsum('ia,ijabc->jbc',p,DK) # term_01
    g_mu_1 = g_mu_1 - np.einsum('ijabcd,jbc->iad',D2K,mu_1) # term_11
    g_p = g_p + np.einsum('jbcd,ijabcd->ia',mu_2,D2K) # term_02
    g_mu_2 = np.einsum('ia,ijabcd->jbcd',p,D2K) # term_02
    g_mu_1 = g_mu_1 + np.einsum('jbcd,ijabecd->iae',mu_2,D3K) # term_12
    g_mu_2 = g_mu_2 + np.einsum('iae,ijabecd->jbcd',mu_1,D3K) # term_12
    g_mu_2 = g_mu_2 + np.einsum('jbcd,ijabcdef->iaef',mu_2,D4K) # term_22
    return np.hstack((g_p.flatten(),g_mu_1.flatten(),g_mu_2.flatten()))

def ode_function_single( x , t ):
    state = x[0:get_dim_state()]

    q , q_1 , q_2,  p , mu_1 , mu_2 = state_to_weinstein_darboux( state )
    K,DK,D2K,D3K,D4K,D5K,D6K = derivatives_of_kernel( q , q )

    dq = np.einsum('ijab,jb->ia',K,p) - np.einsum('ijabc,jbc->ia',DK,mu_1) + np.einsum('jbcd,ijabcd->ia',mu_2,D2K)
    T00 = -np.einsum('ic,jb,ijcba->ia',p,p,DK)
    T01 = np.einsum('id,jbc,ijdbac->ia',p,mu_1,D2K) - np.einsum('jd,ibc,ijdbac->ia',p,mu_1,D2K)
    T02 = -np.einsum('ie,jbcd,ijebacd->ia',p,mu_2,D3K)-np.einsum('je,ibcd,ijebacd->ia',p,mu_2,D3K)
    T12 = -np.einsum('ife,jbcd,ijfbacde->ia',mu_1,mu_2,D4K)+np.einsum('jfe,ibcd,ijfbacde->ia',mu_1,mu_2,D4K)
    T11 = np.einsum('ied,jbc,ijebacd->ia',mu_1,mu_1,D3K)
    T22 = -np.einsum('izef,jbcd,ijzbafcde->ia',mu_2,mu_2,D5K)
    xi_1 = np.einsum('ijacb,jc->iab',DK,p) \
        - np.einsum('ijadbc,jdc->iab',D2K,mu_1) \
        + np.einsum('jecd,ijaebcd->iab',mu_2,D3K)
    xi_2 = np.einsum('ijadbc,jd->iabc',D2K,p) \
        - np.einsum('ijadebc,jde->iabc',D3K,mu_1) \
        + np.einsum('jefd,ijaebcfd->iabc',mu_2,D4K)
    dq_1 = np.einsum('iac,icb->iab',xi_1,q_1)
    dq_2 = np.einsum('iade,idb,iec->iabc',xi_2,q_1,q_1) + np.einsum('iad,idbc->iabc',xi_1,q_2)
    dp = T00 + T01 + T02 + T12 + T11 + T22
    dmu_1 = np.einsum('iac,ibc->iab',mu_1,xi_1)\
        - np.einsum('icb,ica->iab',mu_1,xi_1)\
        + np.einsum('iadc,ibdc->iab',mu_2,xi_2)\
        - np.einsum('idbc,idac->iab',mu_2,xi_2)\
        - np.einsum('idcb,idca->iab',mu_2,xi_2)
    dmu_2 = np.einsum('iadc,ibd->iabc',mu_2,xi_1)\
        + np.einsum('iabd,icd->iabc',mu_2,xi_1)\
        - np.einsum('idbc,ida->iabc',mu_2,xi_1)
    dstate = weinstein_darboux_to_state( dq , dq_1, dq_2, dp , dmu_1 , dmu_2 )

    return dstate

#tmpq = None

def state_to_weinstein_darboux( state ):
    i = 0
    q = np.reshape( state[i:(i+N*DIM)] , [N,DIM] )
    i = i + N*DIM
    q_1 = np.reshape( state[i:(i+N*DIM*DIM)] , [N,DIM,DIM] )
    i = i + N*DIM*DIM
    q_2 = np.reshape( state[i:(i+N*DIM*DIM*DIM)] , [N,DIM,DIM,DIM] )
    i = i + N*DIM*DIM*DIM
    p = np.reshape( state[i:(i+N*DIM)] , [N,DIM] )
    i = i + N*DIM
    mu_1 = np.reshape( state[i:(i + N*DIM*DIM)] , [N,DIM,DIM] )
    i = i + N*DIM*DIM
    mu_2 = np.reshape( state[i:(i + N*DIM*DIM*DIM)] ,[N,DIM,DIM,DIM] ) 
    return q , q_1 , q_2 , p , mu_1 , mu_2

def weinstein_darboux_to_state( q , q_1, q_2, p , mu_1, mu_2 ):
    state = np.zeros( 2* (N*DIM + N*DIM*DIM + N*DIM*DIM*DIM) )
    i = 0
    state[i:(i+N*DIM)] = np.reshape( q , N*DIM )
    i = i + N*DIM 
    state[i:(i+N*DIM*DIM)] = np.reshape( q_1 , N*DIM*DIM )
    i = i + N*DIM*DIM 
    state[i:(i+N*DIM*DIM*DIM)] = np.reshape( q_2 , N*DIM*DIM*DIM )
    i = i + N*DIM*DIM*DIM
    state[i:(i + N*DIM)] = np.reshape( p , N*DIM )
    i = i + N*DIM
    state[i:(i+N*DIM*DIM)] = np.reshape( mu_1 , N*DIM*DIM)
    i = i + N*DIM*DIM
    state[i:(i+N*DIM*DIM*DIM)] = np.reshape( mu_2 , N*DIM*DIM*DIM ) 
    return state


def state_to_triangular(state):
    # remove superfluous entries arising from symmetri in 2nd order indices
    triuind = np.triu_indices(DIM)

    Mq = state[N*DIM+N*DIM**2:N*DIM+N*DIM**2+N*DIM**3].copy().reshape([N,DIM,DIM,DIM])
    Mmu = state[2*N*DIM+2*N*DIM**2+N*DIM**3:2*N*DIM+2*N*DIM**2+2*N*DIM**3].copy().reshape([N,DIM,DIM,DIM])
    for i,a in itertools.product(range(N),range(DIM)):
        Mq[i,a,:,:] = .5*(Mq[i,a,:,:]+Mq[i,a,:,:].T)
        Mmu[i,a,:,:] = .5*(Mmu[i,a,:,:]+Mmu[i,a,:,:].T)

    triuMq = Mq[:,:,triuind[0],triuind[1]]
    triuMmu = Mmu[:,:,triuind[0],triuind[1]]

    statetriu = np.hstack( (state[0:N*DIM+N*DIM**2],triuMq.flatten(),state[N*DIM+N*DIM**2+N*DIM**3:2*N*DIM+2*N*DIM**2+N*DIM**3],triuMmu.flatten(),) )
    assert(statetriu.size == 2*N*DIM+2*N*DIM**2+2*N*DIM*triuDim())

    return statetriu

def triangular_to_state(statetriu):
    # restore superfluous entries arising from symmetri in 2nd order indices
    triuind = np.triu_indices(DIM)
    triuMq = statetriu[N*DIM+N*DIM**2:N*DIM+N*DIM**2+N*DIM*triuDim()].reshape([N,DIM,triuDim()])
    triuMmu = statetriu[2*N*DIM+2*N*DIM**2+N*DIM*triuDim():2*N*DIM+2*N*DIM**2+2*N*DIM*triuDim()].reshape([N,DIM,triuDim()])
    Mq = np.zeros([N,DIM,DIM,DIM])
    Mmu = np.zeros([N,DIM,DIM,DIM])
    for i,a in itertools.product(range(N),range(DIM)):
        Miaq = np.zeros([DIM,DIM])
        Miaq[triuind[0],triuind[1]] = triuMq[i,a,:]
        Mq[i,a,:,:] = .5*(Miaq+Miaq.T)

        Miamu = np.zeros([DIM,DIM])
        Miamu[triuind[0],triuind[1]] = triuMmu[i,a,:]
        Mmu[i,a,:,:] = .5*(Miamu+Miamu.T)

    state = np.hstack( (statetriu[0:N*DIM+N*DIM**2],Mq.flatten(),statetriu[N*DIM+N*DIM**2+N*DIM*triuDim():2*N*DIM+2*N*DIM**2+N*DIM*triuDim()],Mmu.flatten(),) )
    assert(state.size == 2*N*DIM+2*N*DIM**2+2*N*DIM**3)

    return state

def symmetrize_mu_2( state ):
    q,q_1,q_2,p,mu_1,mu_2 = state_to_weinstein_darboux( state )
    for i in range(0,N):
        for d in range(0,DIM):
            store = mu_2[i,d]
            mu_2[i,d] = 0.5*(store + store.T)

def test_Gaussians( q ):
    h = 1e-9
    G,DG,D2G,D3G,D4G,D5G,D6G = gaussian.derivatives_of_Gaussians(q,q)
    q_a = np.copy(q)
    q_b = np.copy(q)
    q_c = np.copy(q)
    q_d = np.copy(q)
    q_e = np.copy(q)
    q_f = np.copy(q)
    for i in range(0,N):
        for a in range(0,DIM):
            error_max = 0.
            q_a[i,a] = q[i,a]+h
            G_a , DG_a , D2G_a , D3G_a, D4G_a , D5G_a ,D6G_a = gaussian.derivatives_of_Gaussians(q_a, q) 
            for j in range(0,N):
                error = (G_a[i,j] - G[i,j])/h - DG[i,j,a]
                error_max = np.maximum( np.absolute( error ) , error_max )
            print 'max error for DG was ' + str( error_max )
            error_max = 0.
            for b in range(0,DIM):
                q_b[i,b] = q[i,b] + h
                G_b , DG_b , D2G_b , D3G_b , D4G_b , D5G_b , D6G_b = gaussian.derivatives_of_Gaussians( q_b , q ) 
                for j in range(0,N):
                    error = (DG_b[i,j,a] - DG[i,j,a])/h - D2G[i,j,a,b]
                    error_max = np.maximum( np.absolute( error ) , error_max )
                print 'max error for D2G was ' + str( error_max )
                error_max = 0.
                for c in range(0,DIM):
                    q_c[i,c] = q_c[i,c] + h
                    G_c , DG_c , D2G_c , D3G_c , D4G_c , D5G_c , D6G_c = gaussian.derivatives_of_Gaussians( q_c , q ) 
                    for j in range(0,N):
                        error = (D2G_c[i,j,a,b] - D2G[i,j,a,b])/h - D3G[i,j,a,b,c]
                        error_max = np.maximum( np.absolute(error) , error_max )
                    print 'max error for D3G was ' + str( error_max )
                    error_max = 0.
                    for d in range(0,DIM):
                        q_d[i,d] = q[i,d] + h
                        G_d, DG_d , D2G_d , D3G_d, D4G_d , D5G_d , D6G_d = gaussian.derivatives_of_Gaussians( q_d , q )
                        for j in range(0,N):
                            error = (D3G_d[i,j,a,b,c] - D3G[i,j,a,b,c])/h - D4G[i,j,a,b,c,d]
                            error_max = np.maximum( np.absolute(error) , error_max )
                        print 'max error for D4G was '+ str(error_max)
                        error_max = 0.
                        for e in range(0,DIM):
                            q_e[i,e] = q[i,e] + h
                            G_e, DG_e , D2G_e , D3G_e, D4G_e, D5G_e , D6G_e = gaussian.derivatives_of_Gaussians( q_e , q )
                            for j in range(0,N):
                                error = (D4G_e[i,j,a,b,c,d] - D4G[i,j,a,b,c,d])/h - D5G[i,j,a,b,c,d,e]
                                error_max = np.maximum( np.absolute(error) , error_max )
                            print 'max error for D5G was '+ str(error_max)
                            for f in range(0,DIM):
                                q_f[i,f] = q[i,f] + h
                                G_f, DG_f , D2G_f , D3G_f, D4G_f, D5G_f,D6G_f = gaussian.derivatives_of_Gaussians( q_f , q )
                                for j in range(0,N):
                                    error = (D5G_f[i,j,a,b,c,d,e] - D5G[i,j,a,b,c,d,e])/h - D6G[i,j,a,b,c,d,e,f]
                                    error_max = np.maximum( np.absolute(error) , error_max )
                                    print 'max error for D6G was '+ str(error_max)
                                    error_max = 0.
                                q_f[i,f] = q_f[i,f] - h
                            q_e[i,e] = q_e[i,e] - h
                        q_d[i,d] = q_d[i,d] - h
                    q_c[i,c] = q_c[i,c] - h
                q_b[i,b] = q_b[i,b] - h
            q_a[i,a] = q_a[i,a] - h
    return 1

def test_kernel_functions( q ):
    h = 1e-8
#    G,DG,D2G,D3G,D4G,D5G = gaussian.derivatives_of_Gaussians(q,q)
    q_a = np.copy(q)
    q_b = np.copy(q)
    q_c = np.copy(q)
    q_d = np.copy(q)
    q_e = np.copy(q)
    K,DK,D2K,D3K,D4K,D5K,D6K = derivatives_of_kernel(q,q)
    delta = np.identity(DIM)
    error_max = 0.
    for i in range(0,N):
        for j in range(0,N):
            x = q[i,:] - q[j,:]
            r_sq = np.inner( x , x )
            for a in range(0,DIM):
                for b in range(0,DIM):
                    G = np.exp( -r_sq / (2.*SIGMA**2) )
                    K_ij_ab = G*delta[a,b]
                    error = K_ij_ab - K[i,j,a,b]
                    error_max = np.maximum( np.absolute(error) , error_max )

    print 'error_max for K = ' + str( error_max )
    if (error_max > 100*h):
        print 'WARNING:  COMPUTATION OF K APPEARS TO BE INACCURATE'

    error_max = 0.
    for i in range(0,N):
        for a in range(0,DIM):
            q_a[i,a] = q[i,a] + h
            K_a,DK_a,D2K_a,D3K_a,D4K_a,D5K_a,D6K_a = derivatives_of_kernel(q_a,q)
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
                K_b,DK_b,D2K_b,D3K_b,D4K_b,D5K_b,D6K_b = derivatives_of_kernel(q_b,q)
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
                    K_c,DK_c,D2K_c,D3K_c,D4K_c,D5K_c,D6K_c = derivatives_of_kernel(q_c,q)
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
    return 1


def test_functions( trials ):
    #checks that each function does what it is supposed to
    h = 10e-7
    q = SIGMA*np.random.randn(N,DIM)
    q1 = SIGMA*np.random.randn(N,DIM,DIM)
    q2 = SIGMA*np.random.randn(N,DIM,DIM,DIM)
    p = SIGMA*np.random.randn(N,DIM)
    mu_1 = np.random.randn(N,DIM,DIM)
#    mu_1 = np.zeros([N,DIM,DIM])
#    mu_2 = np.zeros([N,DIM,DIM,DIM])
    mu_2 = np.random.randn(N,DIM,DIM,DIM)
    
    test_Gaussians( q )
    test_kernel_functions( q )

    s = weinstein_darboux_to_state( q , q1 , q2 ,  p , mu_1 , mu_2 )
    ds = ode_function( s , 0 )
    dq,dq1,dq2,dp_coded,dmu_1,dmu_2 = state_to_weinstein_darboux( ds ) 

    print 'a test of the ode:'
    print 'dp_coded =' + str(dp_coded)
    Q = np.copy(q)
    dp_estim = np.zeros([N,DIM])
    for i in range(0,N):
        for a in range(0,DIM):
            Q[i,a] = q[i,a] + h
            dp_estim[i,a] = - ( Hamiltonian(Q,p,mu_1,mu_2) - Hamiltonian(q,p,mu_1,mu_2) ) / h 
            Q[i,a] = Q[i,a] - h
    print 'dp_estim =' + str(dp_estim)
    print 'dp_error =' + str(dp_estim - dp_coded)
    return 1

test_functions(1)
