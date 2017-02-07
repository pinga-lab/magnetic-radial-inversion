### Functions for magnetic problems with polygonal prisms

import numpy as np
from fatiando import mesher, gridder, utils
from fatiando.gravmag import polyprism
from fatiando.mesher import PolygonalPrism
from fatiando.constants import CM, T2NT

### Functions for the foward problem using fatiando

def pol2cart(m, Np, Nv):
    '''
    This function transforms polar coordinates of the prisms
    into Cartesian coordinates and returns a list of polygonal
    prisms of the Fatiando a Terra.
    
    input
    
    m: list - each element is a list of [r, x0, y0, z1, z2, 'magnetization'],
              whrere r is an array with the radial distances of the vertices,
              x0 and y0 are the origin cartesian coordinates of each prism,
              z1 and z2 are the top and bottom of each prism and
              magnetization is physical property
    Np: int - number of prisms
    Nv: int - number of vertices per prism
    
    output
    
    mk: list - list of objects of the class
    fatiando.mesher.PolygonalPrism
    '''
    
    mk = []
    r = np.zeros(Nv)  # it contains radial distances of the vertices in polar coordinates
    verts = [] # it contains radial distances of the vertices in Cartesian coordinates
 
    assert len(m) == Np, 'The size of m and the number of prisms must be equal'
    
    for mv in m:
        assert len(mv[0]) == Nv, 'All prisms must have Nv vertices'
      
    ang = 2*np.pi/Nv # angle between two vertices

    for mv in m:
        r = mv[0]
        verts=[]
        for i in range(Nv):
            verts.append([r[i]*np.cos(i*ang) + mv[1], r[i]*np.sin(i*ang) + mv[2]])
        mk.append(PolygonalPrism(verts, mv[3], mv[4], mv[5]))
        
    return mk

def param_vec(m, Np, Nv):
    '''
    This function receives the model of prisms and returns the vector of parameters
    
    input
    
    m: list - each element is a list of [r, x0, y0, z1, z2, 'magnetization'],
              whrere r is an array with the radial distances of the vertices,
              x0 and y0 are the origin cartesian coordinates of each prism,
              z1 and z2 are the top and bottom of each prism and
              magnetization is physical property
    Np: int - number of prisms
    Nv: int - number of vertices per prism
    
    output
    
    pv: 1D array - parameters vector
    '''
    
    pv = np.zeros(0) # parameters vector
    mv = [] # list for the loop of asserts
 
    assert len(m) == Np, 'The size of m and the number of prisms must be equal'
    
    for mv in m:
        assert len(mv[0]) == Nv, 'All prisms must have Nv vertices'
    
    for i in range(Np):
        pv = np.hstack((pv, m[i][0], m[i][1:3]))
    
    return pv

### Functions for the derivatives with finite differences

def fd_tf_x0_polyprism(xp, yp, zp, m, Nv, delta, inc, dec):
    '''
    This function calculates the derivative for total field anomaly
    from a model of polygonal prisms using finite difference.
    
    input
    
    xp: array - x observation points
    yp: array - y observation points
    zp: array - z observation points
    m: list - [r, x0, y0, z1, z2, 'magnetization'],
              whrere r is an array with the radial distances of the vertices,
              x0 and y0 are the origin cartesian coordinates of each prism,
              z1 and z2 are the top and bottom of each prism and
              magnetization is the physical property
    Nv: int - number of vertices per prism
    delta: float - increment in x coordinate in meters
    inc: float - inclination
    dec: declination
    
    output
    
    df: 1D array - derivative
    '''
    assert xp.size == yp.size == zp.size, 'The number of points in x, y and z must be equal'
    assert m[0].size + len(m[1:]) == Nv + 5, 'The number of parameter must be Nv + 5'
    
    mp = []  # m + delta
    mm = []  # m - delta
    mp_fat = [] # list of objects of the class fatiando.mesher.PolygonalPrism
    mm_fat = [] # list of objects of the class fatiando.mesher.PolygonalPrism    
    df = np.zeros(xp.size) # derivative
    
    mp = [[m[0], m[1] + delta, m[2], m[3], m[4], m[5]]]
    mm = [[m[0], m[1] - delta, m[2], m[3], m[4], m[5]]]
    
    mp_fat = pol2cart(mp, 1, Nv)
    mm_fat = pol2cart(mm, 1, Nv)
    
    df = (polyprism.tf(xp, yp, zp, mp_fat, inc, dec)\
          - polyprism.tf(xp, yp, zp, mm_fat, inc, dec))
    
    df /= (2.*delta)
    
    return df


def fd_tf_y0_polyprism(xp, yp, zp, m, Nv, delta, inc, dec):
    '''
    This function calculates the derivative for total field anomaly
    from a model of polygonal prisms using finite difference.
    
    input
    
    xp: array - x observation points
    yp: array - y observation points
    zp: array - z observation points
    m: list - [r, x0, y0, z1, z2, 'magnetization'],
              whrere r is an array with the radial distances of the vertices,
              x0 and y0 are the origin cartesian coordinates of each prism,
              z1 and z2 are the top and bottom of each prism and
              magnetization is the physical property
    Nv: int - number of vertices per prism
    delta: float - increment in y coordinate in meters
    inc: float - inclination
    dec: declination
    
    output
    
    df: 1D array - derivative
    '''
    assert xp.size == yp.size == zp.size, 'The number of points in x, y and z must be equal'
    assert len(m[0]) + len(m[1:]) == Nv + 5, 'The number of parameter must be Nv + 5'
    
    mp = []  # m + delta
    mm = []  # m - delta
    mp_fat = [] # list of objects of the class fatiando.mesher.PolygonalPrism
    mm_fat = [] # list of objects of the class fatiando.mesher.PolygonalPrism    
    df = np.zeros(xp.size) # derivative
    
    mp = [[m[0], m[1], m[2] + delta, m[3], m[4], m[5]]]
    mm = [[m[0], m[1], m[2] - delta, m[3], m[4], m[5]]]
    
    mp_fat = pol2cart(mp, 1, Nv)
    mm_fat = pol2cart(mm, 1, Nv)
    
    df = (polyprism.tf(xp, yp, zp, mp_fat, inc, dec)\
          - polyprism.tf(xp, yp, zp, mm_fat, inc, dec))
    
    df /= (2.*delta)
    
    return df

def fd_tf_radial_polyprism(xp, yp, zp, m, Nv, nv, delta, inc, dec):
    '''
    This function calculates the derivative for total field anomaly
    from a model of polygonal prisms using finite difference.
    
    input
    
    xp: array - x observation points
    yp: array - y observation points
    zp: array - z observation points
    m: list - [r, x0, y0, z1, z2, 'magnetization'],
              whrere r is an array with the radial distances of the vertices,
              x0 and y0 are the origin cartesian coordinates of each prism,
              z1 and z2 are the top and bottom of each prism and
              magnetization is the physical property
    Nv: int - number of vertices per prism
    nv: int - number of the vertice for the derivative
    delta: float - increment in radial distance in meters
    inc: float - inclination
    dec: declination
    
    output
    
    df: 1D array - derivative
    '''
    assert xp.size == yp.size == zp.size, 'The number of points in x, y and z must be equal'
    assert len(m[0]) + len(m[1:]) == Nv + 5, 'The number of parameter must be Nv + 5'
    assert nv < Nv, 'The vertice number must be smaller than the number of vertices (0 - Nv)'
    
    m_fat = [] # list of objects of the class fatiando.mesher.PolygonalPrism
    df = np.zeros(xp.size) # derivative
    verts = [] # vertices of new prism
    ang = 2.*np.pi/Nv # angle between two vertices
    
    if nv == Nv - 1:
        nvp = 0
    else:
        nvp = nv + 1
        
    cos_nvm = np.cos((nv - 1)*ang)
    sin_nvm = np.sin((nv - 1)*ang)
    cos_nv = np.cos(nv*ang)
    sin_nv = np.sin(nv*ang)
    cos_nvp = np.cos(nvp*ang)
    sin_nvp = np.sin(nvp*ang)
    
    verts.append([m[0][nv - 1]*cos_nvm, m[0][nv - 1]*sin_nv])
    verts.append([(m[0][nv] + delta)*cos_nv, (m[0][nv] + delta)*sin_nv])
    verts.append([m[0][nvp]*cos_nvp, m[0][nvp]*sin_nvp])
    verts.append([(m[0][nv] - delta)*cos_nv, (m[0][nv] - delta)*sin_nv])

    m_fat = [PolygonalPrism(verts, m[3], m[4], m[5])]
    
    df = polyprism.tf(xp, yp, zp, m_fat, inc, dec)
    df /= (2.*delta)
    
    return df

def fd_tf_sm_polyprism(xp, yp, zp, m, Np, Nv, deltax, deltay, deltar, inc, dec):
    '''
    This function calculates the derivative for total field anomaly
    from a model of polygonal prisms using finite difference.
    
    input
    
    xp: array - x observation points
    yp: array - y observation points
    zp: array - z observation points
    m: list - each element is a list of [r, x0, y0, z1, z2, 'magnetization'],
              where r is an array with the radial distances of the vertices,
              x0 and y0 are the origin cartesian coordinates of each prism,
              z1 and z2 are the top and bottom of each prism and
              magnetization is the physical property
    Np: int - number of prisms
    Nv: int - number of vertices per prism
    deltax: float - increment in x coordinate in meters
    deltay: float - increment in y coordinate in meters
    deltar: float - increment in z coordinate in meters
    inc: float - inclination of the local-geomagnetic field
    dec: declination of the local-geomagnetic field
    
    output
    
    G: 2D array - sensibility matrix
    '''
    for mv in m:
        assert len(mv[0]) == Nv, 'All prisms must have Nv vertices'
    assert xp.size == yp.size == zp.size, 'The number of points in x, y and z must be equal'
    
    pp = 2 + Nv # number of parameters per prism
    
    G = np.zeros((xp.size, pp*Np))
    
    for i, mv in enumerate(m):
        aux = i*pp
        G[:, aux + Nv] = fd_tf_x0_polyprism(xp, yp, zp, mv, Nv, deltax, inc, dec)
        G[:, aux + Nv + 1] = fd_tf_y0_polyprism(xp, yp, zp, mv, Nv, deltay, inc, dec)
        for j in range(Nv):
            G[:, aux + j] = fd_tf_radial_polyprism(xp, yp, zp, mv, Nv, j, deltar, inc, dec)
            
    return G

### Functions for the inversion constraints

def Hessian_phi_1(M, L, H, alpha):
    '''
    Returns the hessian matrix constrained by smoothness constraint
    on the adjacent radial distances within each prism.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    H: 2D array - hessian matrix
    alpha: float - weight
       
    output
    
    H: 2D array - hessian matrix plus phi_1 constraint
    '''
    
    P = L*(M + 2)
    
    assert H.shape == (P, P), 'The hessian shape must be (P, P)'
    
    # extracting the non-zero diagonals
    d0, d1, dM = diags_phi_1(M, L, alpha)
    
    i, j = np.diag_indices_from(H) # indices of the diagonal elements
    
    k = np.full(P-1, 1, dtype=np.int) # array iterable
    l = np.full(P-M+1, M-1, dtype=np.int) # array iterable    
    
    H[i,j] += d0
    H[i[:P-1],j[:P-1] + k] += d1
    H[i[:P-1] + k,j[:P-1]] += d1
    H[i[:P-M+1],j[:P-M+1] + l] += dM
    H[i[:P-M+1] + l,j[:P-M+1]] += dM
    
    return H

def Hessian_phi_2(M, L, H, alpha):
    '''
    Returns the hessian matrix constrained by smoothness constraint
    on radial distances of the vertically adjacent prisms.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    H: 2D array - hessian matrix
    alpha: float - weight
       
    output
    
    H: 2D array - hessian matrix plus phi_2 constraint
    '''
    
    P = L*(M + 2)
    
    assert H.shape == (P, P), 'The hessian shape must be (P, P)'
    
    # extracting the non-zero diagonals
    d0, d1 = diags_phi_2(M, L, alpha)
    
    i, j = np.diag_indices_from(H) # indices of the diagonal elements
    
    k = np.full(P-M-2, M+2, dtype=np.int) # array iterable
    
    H[i,j] += d0
    H[i[:P-M-2],j[:P-M-2] + k] += d1
    H[i[:P-M-2] + k,j[:P-M-2]] += d1
    
    return H

def Hessian_phi_3(M, L, H, alpha):
    '''
    Returns the hessian matrix constrained that the estimated cross-section
    of the shallowest prism must be close to the known outcropping boundary.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    H: 2D array - hessian matrix
    alpha: float - weight
       
    output
    
    H: 2D array - hessian matrix plus phi_3 constraint
    '''
    
    P = L*(M + 2)
    
    assert H.shape == (P, P), 'The hessian shape must be (P, P)'
    
    i, j = np.diag_indices(M+2) # indices of the diagonal elements in M + 2
    
    H[i,j] += alpha
    
    return H

def Hessian_phi_4(M, L, H, alpha):
    '''
    Returns the hessian matrix constrained that the estimated origin
    of the shallowest prism must be close to the known outcropping origin.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    H: 2D array - hessian matrix
    alpha: float - weight
       
    output
    
    H: 2D array - hessian matrix plus phi_4 constraint
    '''
    
    P = L*(M + 2)
    
    assert H.shape == (P, P), 'The hessian shape must be (P, P)'
    
    i, j = np.diag_indices(P) # indices of the diagonal elements
    
    H[M,M] += alpha
    H[M+1,M+1] += alpha
    
    return H

def Hessian_phi_5(M, L, H, alpha):
    '''
    Returns the hessian matrix constrained by smoothness constraint
    on the origins vertically adjacent prisms.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    H: 2D array - hessian matrix
    alpha: float - weight
       
    output
    
    H: 2D array - hessian matrix plus phi_5 constraint
    '''
    
    P = L*(M + 2)
    
    assert H.shape == (P, P), 'The hessian shape must be (P, P)'
    
    # extracting the non-zero diagonals
    d0, d1 = diags_phi_5(M, L, alpha)
    
    i, j = np.diag_indices_from(H) # indices of the diagonal elements
    
    k = np.full(P-1, 1, dtype=np.int) # array iterable
    l = np.full(P-M-2, M+2, dtype=np.int) # array iterable
    
    H[i,j] += d0
    H[i[:P-M-2],j[:P-M-2] + l] += d1
    H[i[:P-M-2] + l,j[:P-M-2]] += d1
    
    return H

def Hessian_phi_6(M, L, H, alpha):
    '''
    Returns the hessian matrix constrained that radial distances
    within each prism must be close to null values.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    H: 2D array - hessian matrix
    alpha: float - weight
       
    output
    
    H: 2D array - hessian matrix plus phi_6 constraint
    '''
    
    P = L*(M + 2)
    
    assert H.shape == (P, P), 'The hessian shape must be (P, P)'
    
    # extracting the non-zero diagonals
    d0 = diags_phi_6(M, L, alpha)
    
    i, j = np.diag_indices_from(H) # indices of the diagonal elements
    
    H[i,j] += d0
    
    return H

def gradient_phi_1(M, L, m, alpha):
    '''
    Returns the gradient vector constrained by smoothness constraint
    on the adjacent radial distances within each prism.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    m: 1D array - gradient of parameter vector
    alpha: float - weight
    
    output
    
    m: 1D array - gradient vector plus phi_1 constraint
    '''
    
    P = L*(M + 2)
    
    assert m.size == P, 'The size of parameter vector must be equal to P'
    
    m1 = m # the new vector m1 = gradient input + gradient of phi1
    
    # extracting the non-zero diagonals
    d0, d1, dM = diags_phi_1(M, L, alpha)
    
    # calculating the product between the diagonals and the slices of m
    m1 += m*d0
    m1[1:] += m[1:]*d1
    m1[:P-1] += m[:P-1]*d1
    m1[M-1:] += m[M-1:]*dM
    m1[:P-M+1] += m[:P-M+1]*dM
       
    return m1

def gradient_phi_2(M, L, m, alpha):
    '''
    Returns the gradient vector constrained by smoothness constraint
    on radial distances of the vertically adjacent prisms.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    m: 1D array - gradient of parameter vector
    alpha: float - weight
    
    output
    
    m2: 1D array - gradient vector plus phi_2 constraint
    '''
    
    m2 = m # the new vector m1 = gradient input + gradient of phi2
    
    P = L*(M + 2)
    
    assert m.size == P, 'The size of parameter vector must be equal to P'
    
    # extracting the non-zero diagonals
    d0, d1 = diags_phi_2(M, L, alpha)
    
    # calculating the product between the diagonals and the slices of m
    m2 += m*d0
    m2[M+2:] += m[M+2:]*d1
    m2[:P-M-2] += m[:P-M-2]*d1
    
    return m2

def gradient_phi_3(M, L, m, m0, alpha):
    '''
    Returns the gradient vector constrained that the estimated cross-section
    of the shallowest prism must be close to the known outcropping boundary.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    m: 1D array - gradient of parameter vector
    m0: 1D array - parameters of the outcropping body
    alpha: float - weight
    
    output
    
    m: 1D array - gradient vector plus phi_3 constraint
    '''
    
    P = L*(M + 2)
    
    assert m.size == P, 'The size of parameter vector must be equal to P'
    assert m0.size == M+2, 'The size of parameter vector must be equal to M + 2'

    
    # calculating the product between the diagonals and the slices of m
    m[:M+2] += m[:M+2] - m0
        
    return m

def gradient_phi_4(M, L, m, m0, alpha):
    '''
    Returns the gradient vector constrained that the estimated origin
    of the shallowest prism must be close to the known outcropping origin.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    m: 1D array - gradient of parameter vector
    m0: 1D array - origin (x0,y0) of the outcropping body
    alpha: float - weight
    
    output
    
    m: 1D array - gradient vector plus phi_4 constraint
    '''
    
    P = L*(M + 2)
    
    assert m.size == P, 'The size of parameter vector must be equal to P'
    assert m0.size == 2, 'The size of parameter vector must be equal to 2'
    
    # calculating the product between the diagonals and the slices of m
    m[M:M+2] += m[M:M+2] - m0
    
    return m

def gradient_phi_5(M, L, m, alpha):
    '''
    Returns the gradient vector constrained by smoothness constraint
    on the origins vertically adjacent prisms.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    m: 1D array - gradient of parameter vector
    alpha: float - weight
    
    output
    
    m5: 1D array - gradient vector plus phi_5 constraint
    '''
    
    m5 = m # the new vector m1 = gradient input + gradient of phi5
    
    P = L*(M + 2)
    
    assert m.size == P, 'The size of parameter vector must be equal to P'
    
    # extracting the non-zero diagonals
    d0, d1 = diags_phi_5(M, L, alpha)
    
    # calculating the product between the diagonals and the slices of m
    m5 += m*d0
    m5[M+2:] += m[M+2:]*d1
    m5[:P-M-2] += m[:P-M-2]*d1
    
    return m5

def gradient_phi_6(M, L, m, alpha):
    '''
    Returns the gradient vector constrained that radial distances
    within each prism must be close to null values.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    m: 1D array - gradient of parameter vector
    alpha: float - weight
    
    output
    
    m: 1D array - gradient vector plus phi_6 constraint
    '''
    
    P = L*(M + 2)
    
    assert m.size == P, 'The size of parameter vector must be equal to P'
    
    # extracting the non-zero diagonals
    d0 = diags_phi_6(M, L, alpha)
    
    # calculating the product between the diagonals and the slices of m
    m += m*d0
    
    return m

def diags_phi_1(M, L, alpha):
    '''
    Returns the non-zero diagonals of hessian matrix for 
    the smoothness constraint on adjacent radial distances
    in the same prism.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    alpha: float - weight
    
    output
    
    d0, d1, dM: 1D array - diagonals from phi_1 hessian
    '''
    
    P = L*(M + 2)
    
    # building the diagonals
    d0 = np.zeros(M+2)
    d0[:M] = 2.*alpha
    d0 = np.resize(d0, P)
    
    d1 = np.zeros(M+2)
    d1[:M-1] = - alpha
    d1 = np.resize(d1, P-1)
    
    dM = np.zeros(M+2)
    dM[0] = - alpha
    dM = np.resize(dM, P - M + 1)
    
    return d0, d1, dM

def diags_phi_2(M, L, alpha):
    '''
    Returns the non-zero diagonals of hessian matrix for 
    the smoothness constraint on adjacent radial distances
    in the adjacent prisms.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    alpha: float - weight
    
    output
    
    d0, d1: 1D array - diagonals from phi_2 hessian
    '''
    
    P = L*(M + 2)
    
    # building the diagonals
    
    if M <= 2:
        d0 = np.zeros(M+2)
        d0[:M] = alpha
        d0 = np.resize(d0, P)
    else:
        d0 = np.zeros(M+2)
        d0[:M] = 2*alpha
        d0 = np.resize(d0, P)
        d0[:M] = alpha
        d0[-M-2:-M+1] = alpha        
    
    d1 = np.zeros(M+2)
    d1[:M] = - alpha
    d1 = np.resize(d1, P-M-2)
    
    return d0, d1

def diags_phi_3(M, L, alpha):
    '''
    Returns the non-zero diagonals of hessian matrix for 
    the smoothness constraint on adjacent radial distances
    in the adjacent prisms.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    alpha: float - weight
    
    output
    
    d0, d1: 1D array - diagonals from phi_3 hessian
    '''
    
    P = L*(M + 2)
    
    # building the diagonals
    
    if M <= 2:
        d0 = np.zeros(M+2)
        d0[:M] = alpha
        d0 = np.resize(d0, P)
    else:
        d0 = np.zeros(M+2)
        d0[:M] = 2*alpha
        d0 = np.resize(d0, P)
        d0[:M] = alpha
        d0[-M-2:-M+1] = alpha        
    
    d1 = np.zeros(M+2)
    d1[:M+2] = - alpha
    d1 = np.resize(d1, P-M-2)
    
    return d0, d1

def diags_phi_5(M, L, alpha):
    '''
    Returns the non-zero diagonals of hessian matrix for 
    the smoothness constraint on origin in the adjacent prisms.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    alpha: float - weight
    
    output
    
    d0, d1: 1D array - diagonals from phi_5 hessian
    '''
    
    P = L*(M + 2)
    
    # building the diagonals
    
    if M <= 2:
        d0 = np.zeros(M+2)
        d0[M:M+2] = alpha
        d0 = np.resize(d0, P)
    else:
        d0 = np.zeros(M+2)
        d0[M:M+2] = 2*alpha
        d0 = np.resize(d0, P)
        d0[M:M+2] = alpha
        d0[-2:] = alpha        
    
    d1 = np.zeros(M+2)
    d1[M:M+2] = - alpha
    d1 = np.resize(d1, P-M-2)
    
    return d0, d1

def diags_phi_6(M, L, alpha):
    '''
    Returns the non-zero diagonals of hessian matrix for 
    an minimum Euclidian norm on adjacent radial distances
    within each prisms.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    alpha: float - weight
    
    output
    
    d0: 1D array - diagonal from phi_6 hessian
    '''
    
    P = L*(M + 2)
    
    # building the diagonal
    d0 = np.zeros(M+2)
    d0[:M] = alpha
    d0 = np.resize(d0, P)
    
    return d0