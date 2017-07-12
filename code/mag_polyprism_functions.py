### Functions for magnetic problems with polygonal prisms

import numpy as np
from fatiando import mesher, gridder, utils
from fatiando.gravmag import polyprism
from fatiando.mesher import PolygonalPrism
from fatiando.constants import CM, T2NT
from copy import deepcopy

### Functions for the foward problem using fatiando

def area_polygon(x, y):
    '''
    Returns the area of a polygon using the shoelace
    formula.
    
    input
    
    x: 1D array - Cartesian coordinates
    y: 1D array - Cartesian coordinates
    
    output
    
    area: float - area of the polygon
    '''
    
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    n = len(x)
    shift_up = np.arange(-n+1, 1)
    shift_down = np.arange(-1, n-1)
    area = (x * (y.take(shift_up) - y.take(shift_down))).sum() / 2.0
    return area

def pol2cart(m, M, L):
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
    M: int - number of vertices per prism
    L: int - number of prisms
    
    output
    
    mk: list - list of objects of the class
    fatiando.mesher.PolygonalPrism
    '''
    
    mk = []
    r = np.zeros(M)  # it contains radial distances of the vertices in polar coordinates
    verts = [] # it contains radial distances of the vertices in Cartesian coordinates
 
    assert len(m) == L, 'The size of m and the number of prisms must be equal'
    
    for mv in m:
        assert len(mv[0]) == M, 'All prisms must have M vertices'
      
    ang = 2*np.pi/M # angle between two vertices

    for mv in m:
        r = mv[0]
        verts=[]
        for i in range(M):
            verts.append([r[i]*np.cos(i*ang) + mv[1], r[i]*np.sin(i*ang) + mv[2]])
        mk.append(PolygonalPrism(verts, mv[3], mv[4], mv[5]))
        
    return mk

def param_vec(m, M, L):
    '''
    This function receives the model of prisms and returns the vector of parameters
    
    input
    
    m: list - each element is a list of [r, x0, y0, z1, z2, 'magnetization'],
              whrere r is an array with the radial distances of the vertices,
              x0 and y0 are the origin cartesian coordinates of each prism,
              z1 and z2 are the top and bottom of each prism and
              magnetization is physical property
    M: int - number of vertices per prism
    L: int - number of prisms
    
    output
    
    pv: 1D array - parameters vector
    '''
    
    pv = np.zeros(0) # parameters vector
    mv = [] # list for the loop of asserts
 
    assert len(m) == L, 'The size of m and the number of prisms must be equal'
    
    for mv in m:
        assert len(mv[0]) == M, 'All prisms must have M vertices'
    
    for i in range(L):
        pv = np.hstack((pv, m[i][0], m[i][1:3]))
    
    return pv

def param2model(m, M, L, z0, dz, props):
    '''
    Returns a model of list of objects of the class
    fatiando.mesher.PolygonalPrism
    
    input
    
    m: 1D array - parameter vector
    M: int - number of vertices
    L: int - number of prisms
    z0: float - top of the model
    dz: float - thickness of each prism
    props: dictionary - physical property
    
    output
    
    mv: list - list of prisms 
    '''
    P = L*(M + 2)
    assert m.size == P, 'The size of m must be equal to L*(M + 2)'
    
    r = np.zeros(M) # vector for radial distances
    mv = [] # list of prisms    
    
    k = 0.
    for i in range(0, P, M + 2):
        r = m[i:M+i]
        mv.append([r, m[i+M], m[i+M+1], z0 + dz*k, z0 + dz*(k + 1.), props])
        k = k + 1.
        
    return mv

### Functions for the derivatives with finite differences

def fd_tf_x0_polyprism(xp, yp, zp, m, M, delta, inc, dec):
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
    M: int - number of vertices per prism
    delta: float - increment in x coordinate in meters
    inc: float - inclination
    dec: declination
    
    output
    
    df: 1D array - derivative
    '''
    assert xp.size == yp.size == zp.size, 'The number of points in x, y and z must be equal'
    assert m[0].size + len(m[1:]) == M + 5, 'The number of parameter must be M + 5'
    
    mp = []  # m + delta
    mm = []  # m - delta
    mp_fat = [] # list of objects of the class fatiando.mesher.PolygonalPrism
    mm_fat = [] # list of objects of the class fatiando.mesher.PolygonalPrism    
    df = np.zeros(xp.size) # derivative
    
    mp = [[m[0], m[1] + delta, m[2], m[3], m[4], m[5]]]
    mm = [[m[0], m[1] - delta, m[2], m[3], m[4], m[5]]]
    
    mp_fat = pol2cart(mp, M, 1)
    mm_fat = pol2cart(mm, M, 1)
    
    df = polyprism.tf(xp, yp, zp, mp_fat, inc, dec)
    df -= polyprism.tf(xp, yp, zp, mm_fat, inc, dec)
    
    df /= (2.*delta)
    
    return df


def fd_tf_y0_polyprism(xp, yp, zp, m, M, delta, inc, dec):
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
    M: int - number of vertices per prism
    delta: float - increment in y coordinate in meters
    inc: float - inclination
    dec: declination
    
    output
    
    df: 1D array - derivative
    '''
    assert xp.size == yp.size == zp.size, 'The number of points in x, y and z must be equal'
    assert len(m[0]) + len(m[1:]) == M + 5, 'The number of parameter must be M + 5'
    
    mp = []  # m + delta
    mm = []  # m - delta
    mp_fat = [] # list of objects of the class fatiando.mesher.PolygonalPrism
    mm_fat = [] # list of objects of the class fatiando.mesher.PolygonalPrism    
    df = np.zeros(xp.size) # derivative
    
    mp = [[m[0], m[1], m[2] + delta, m[3], m[4], m[5]]]
    mm = [[m[0], m[1], m[2] - delta, m[3], m[4], m[5]]]
    
    mp_fat = pol2cart(mp, M, 1)
    mm_fat = pol2cart(mm, M, 1)
    
    df = polyprism.tf(xp, yp, zp, mp_fat, inc, dec)
    df -= polyprism.tf(xp, yp, zp, mm_fat, inc, dec)
    
    df /= (2.*delta)
    
    return df

def fd_tf_radial_polyprism(xp, yp, zp, m, M, nv, delta, inc, dec):
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
    M: int - number of vertices per prism
    nv: int - number of the vertice for the derivative
    delta: float - increment in radial distance in meters
    inc: float - inclination
    dec: declination
    
    output
    
    df: 1D array - derivative
    '''
    assert xp.size == yp.size == zp.size, 'The number of points in x, y and z must be equal'
    assert len(m[0]) + len(m[1:]) == M + 5, 'The number of parameter must be M + 5'
    assert nv < M, 'The vertice number must be smaller than the number of vertices (0 - M)'
    
    m_fat = [] # list of objects of the class fatiando.mesher.PolygonalPrism
    df = np.zeros(xp.size) # derivative
    verts = [] # vertices of new prism
    ang = 2.*np.pi/M # angle between two vertices
    
    if nv == M - 1:
        nvp = 0
    else:
        nvp = nv + 1
        
    cos_nvm = np.cos((nv - 1)*ang)
    sin_nvm = np.sin((nv - 1)*ang)
    cos_nv = np.cos(nv*ang)
    sin_nv = np.sin(nv*ang)
    cos_nvp = np.cos(nvp*ang)
    sin_nvp = np.sin(nvp*ang)
    
    verts.append([m[0][nv - 1]*cos_nvm + m[1], m[0][nv - 1]*sin_nvm + m[2]])
    verts.append([(m[0][nv] + delta)*cos_nv + m[1], (m[0][nv] + delta)*sin_nv + m[2]])
    verts.append([m[0][nvp]*cos_nvp + m[1], m[0][nvp]*sin_nvp + m[2]])
    verts.append([(m[0][nv] - delta)*cos_nv + m[1], (m[0][nv] - delta)*sin_nv + m[2]])

    m_fat = [PolygonalPrism(verts, m[3], m[4], m[5])]
    
    df = polyprism.tf(xp, yp, zp, m_fat, inc, dec)
    df /= (2.*delta)
    
    return df

def fd_tf_sm_polyprism(xp, yp, zp, m, M, L, deltax, deltay, deltar, inc, dec):
    '''
    Returns the sensibility matrix for polygonal prisms using finite 
    differences.
    
    input
    
    xp: array - x observation points
    yp: array - y observation points
    zp: array - z observation points
    m: list - each element is a list of [r, x0, y0, z1, z2, 'magnetization'],
              where r is an array with the radial distances of the vertices,
              x0 and y0 are the origin cartesian coordinates of each prism,
              z1 and z2 are the top and bottom of each prism and
              magnetization is the physical property
    M: int - number of vertices per prism
    L: int - number of prisms
    deltax: float - increment in x coordinate in meters
    deltay: float - increment in y coordinate in meters
    deltar: float - increment in z coordinate in meters
    inc: float - inclination of the local-geomagnetic field
    dec: declination of the local-geomagnetic field
    
    output
    
    G: 2D array - sensibility matrix
    '''
    for mv in m:
        assert len(mv[0]) == M, 'All prisms must have M vertices'
    assert xp.size == yp.size == zp.size, 'The number of points in x, y and z must be equal'
    
    pp = 2 + M # number of parameters per prism
    
    G = np.zeros((xp.size, pp*L))
    
    for i, mv in enumerate(m):
        aux = i*pp
        G[:, aux + M] = fd_tf_x0_polyprism(xp, yp, zp, mv, M, deltax, inc, dec)
        G[:, aux + M + 1] = fd_tf_y0_polyprism(xp, yp, zp, mv, M, deltay, inc, dec)
        for j in range(M):
            G[:, aux + j] = fd_tf_radial_polyprism(xp, yp, zp, mv, M, j, deltar, inc, dec)
            
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
    
    m1 = m.copy() # the new vector m1 = gradient input + gradient of phi1
    
    # extracting the non-zero diagonals
    d0, d1, dM = diags_phi_1(M, L, alpha)
    
    # calculating the product between the diagonals and the slices of m
    m1 += m*d0
    m1[:P-1] += m[1:]*d1
    m1[1:] += m[:P-1]*d1
    m1[:P-M+1] += m[M-1:]*dM
    m1[M-1:] += m[:P-M+1]*dM
       
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
    
    m2 = m.copy() # the new vector m2 = gradient input + gradient of phi2
    
    P = L*(M + 2)
    
    assert m.size == P, 'The size of parameter vector must be equal to P'
    
    # extracting the non-zero diagonals
    d0, d1 = diags_phi_2(M, L, alpha)
    
    # calculating the product between the diagonals and the slices of m
    m2 += m*d0
    m2[:P-M-2] += m[M+2:]*d1
    m2[M+2:] += m[:P-M-2]*d1
    
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
    assert m0.size == M + 2, 'The size of parameter vector must be equal to M + 2'

    
    # calculating the product between the diagonals and the slices of m
    m[:M+2] += (m[:M+2] - m0)*alpha
        
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
    m[M:M+2] += (m[M:M+2] - m0)*alpha
    
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
    
    m5 = m.copy() # the new vector m1 = gradient input + gradient of phi5
    
    P = L*(M + 2)
    
    assert m.size == P, 'The size of parameter vector must be equal to P'
    
    # extracting the non-zero diagonals
    d0, d1 = diags_phi_5(M, L, alpha)
    
    # calculating the product between the diagonals and the slices of m
    m5 += m*d0
    m5[:P-M-2] += m[M+2:]*d1
    m5[M+2:] += m[:P-M-2]*d1
    
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

def phi_1(M, L, m, alpha):
    '''
    Returns the value for the phi1 constraint.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    m: 1D array - parameter vector
    alpha: float - regularization parameter
    
    output
    
    phi_1: float - value of phi_1 constraint
    '''
    
    P = L*(M + 2)
    
    assert m.size == P, 'The size of parameter vector must be equal to P'
    
    # extracting the non-zero diagonals
    d0, d1, dM = diags_phi_1(M, L, alpha)
    
    # calculating the product between the diagonals and the slices of m
    m1 = m*d0
    m1[:P-1] += m[1:]*d1
    m1[1:] += m[:P-1]*d1
    m1[:P-M+1] += m[M-1:]*dM
    m1[M-1:] += m[:P-M+1]*dM
 
    phi_1 = np.dot(m1, m)
    
    return phi_1

def phi_2(M, L, m, alpha):
    '''
    Returns the value for the phi2 constraint.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    m: 1D array - parameter vector
    alpha: float - weight
    
    output
    
    phi_2: float - value of phi_2 constraint
    '''
    
    P = L*(M + 2)
    
    assert m.size == P, 'The size of parameter vector must be equal to P'
    
    # extracting the non-zero diagonals
    d0, d1 = diags_phi_2(M, L, alpha)
    
    # calculating the product between the diagonals and the slices of m
    m2 = m*d0
    m2[:P-M-2] += m[M+2:]*d1
    m2[M+2:] += m[:P-M-2]*d1
    
    phi_2 = np.dot(m2, m)
    
    return phi_2

def phi_3(M, L, m, m0, alpha):
    '''
    Returns the value for the phi3 constraint.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    m: 1D array - parameter vector
    m0: 1D array - parameters of the outcropping body
    alpha: float - weight
    
    output
    
    phi_3: float - value of phi_3 constraint
    '''
    
    P = L*(M + 2)
    
    assert m.size == P, 'The size of parameter vector must be equal to P'
    assert m0.size == M + 2, 'The size of parameter vector must be equal to M + 2'
    
    m3 = np.zeros(P)

    # calculating the product between the diagonals and the slices of m
    m3[:M+2] = (m[:M+2] - m0)*alpha
    
    phi_3 = np.dot(m3, m)
        
    return phi_3

def phi_4(M, L, m, m0, alpha):
    '''
    Returns the value for the phi4 constraint.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    m: 1D array - parameter vector
    m0: 1D array - parameters of the outcropping body
    alpha: float - weight
    
    output
    
    phi_4: float - value of phi_4 constraint
    '''
    
    P = L*(M + 2)
    
    assert m.size == P, 'The size of parameter vector must be equal to P'
    assert m0.size == 2, 'The size of parameter vector must be equal to 2'
    
    m4 = np.zeros(P)

    # calculating the product between the diagonals and the slices of m
    m4[M:M+2] = (m[M:M+2] - m0)*alpha
    
    phi_4 = np.dot(m4, m)
        
    return phi_4

def phi_5(M, L, m, alpha):
    '''
    Returns the value for the phi5 constraint.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    m: 1D array - parameter vector
    alpha: float - weight
    
    output
    
    phi_5: float - value of phi_5 constraint
    '''
    
    P = L*(M + 2)
    
    assert m.size == P, 'The size of parameter vector must be equal to P'
    
    # extracting the non-zero diagonals
    d0, d1 = diags_phi_5(M, L, alpha)
    
    # calculating the product between the diagonals and the slices of m
    m5 = m*d0
    m5[:P-M-2] += m[M+2:]*d1
    m5[M+2:] += m[:P-M-2]*d1
    
    phi_5 = np.dot(m5, m)
    
    return phi_5

def phi_6(M, L, m, alpha):
    '''
    Returns the value for the phi6 constraint.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    m: 1D array - parameter vector
    alpha: float - weight
    
    output
    
    phi_6: float - value of phi_6 constraint
    '''
    
    P = L*(M + 2)
    
    assert m.size == P, 'The size of parameter vector must be equal to P'
    
    # extracting the non-zero diagonals
    d0 = diags_phi_6(M, L, alpha)
    
    # calculating the product between the diagonals and the slices of m
    m6 = m*d0

    phi_6 = np.dot(m6, m)
    
    return phi_6

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
    dM = np.resize(dM, P-M+1)
    
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
    assert L >= 2, 'The number of prisms must be greater than 1'
        
    P = L*(M + 2)
    
    # building the diagonals
    
    if L <= 2:
        d0 = np.zeros(M+2)
        d0[:M] = alpha
        d0 = np.resize(d0, P)
    else:
        d0 = np.zeros(M+2)
        d0[:M] = 2.*alpha
        d0 = np.resize(d0, P)
        d0[:M] -= alpha
        d0[-M-2:-M+2] -= alpha        
    
    d1 = np.zeros(M+2)
    d1[:M] = - alpha
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
    assert L >= 2, 'The number of prisms must be greater than 1'
    
    P = L*(M + 2)
    
    # building the diagonals
    
    if L == 2:
        d0 = np.zeros(M+2)
        d0[M:M+2] = alpha
        d0 = np.resize(d0, P)
    else:
        d0 = np.zeros(M+2)
        d0[M:M+2] = 2*alpha
        d0 = np.resize(d0, P)
        d0[M:M+2] -= alpha
        d0[-2:] -= alpha        
    
    d1 = np.zeros(M+2)
    d1[M:M+2] = -alpha
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

# Functions for inverse problem

def trans_parameter(m, M, L, rmin, rmax, x0min, x0max, y0min, y0max):
    '''
    Returns transformated parameters.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    m: 1D array - parameter vector with radial distances of each vertice
                  and the Cartesian coordinates of each prism
    rmin: float - minimum value of radial distances
    rmax: float - maximum value of radial distances
    x0min: float - minimum value of x coordinate of origins
    x0max: float - maximum value of x coordinate of origins
    y0min: float - minimum value of y coordinate of origins
    y0max: float - maximum value of y coordinate of origins
    
    output
    
    mt: 1D array - transformated parameter vector
    '''
    
    assert len(m) == L*(M + 2), 'The size of m must be equal to L*(M + 2)'
    

    mt = np.zeros_like(m)
    
    P = L*(M+2)  # number of parameters
    
    for i in range(0, P, M+2):
        mt[i:M+i] = - np.log((rmax - m[i:M+i])/(m[i:M+i] - rmin))
        mt[i+M] = - np.log((x0max - m[i+M])/(m[i+M] - x0min))
        mt[i+M+1] = - np.log((y0max - m[i+M+1])/(m[i+M+1] - y0min))
        
    return mt


def trans_parameter2(m, M, L, mmax, mmin):
    '''
    Returns the transformated parameters.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    mt: 1D array - transformated parameters vector with 
                  radial distances of each vertice
                  and the Cartesian coordinates of each prism
    mmax: 1D array - maximum value of each parameter
                    (r1max,...,rMmax, x0max, y0max)
    mmin: 1D array - minimum value of each parameter
                    (r1min,...,rMmin, x0min, y0min)
    
    output
    
    mt: 1D array - parameters vector
    '''
    assert mmax.size == L*(M + 2), 'The size of mmax must be equal to L*(M + 2)'
    assert mmin.size == L*(M + 2), 'The size of mmin must be equal to L*(M + 2)'
    assert m.size == L*(M + 2), 'The size of m must be equal to L*(M + 2)'

    mt = -np.log(abs((mmax - m))/(m - mmin))

    return mt


def trans_inv_parameter(mt, M, L, rmin, rmax, x0min, x0max, y0min, y0max):
    '''
    Returns the parameters from the inverse transformation.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    mt: 1D array - transformated parameters vector with 
                  radial distances of each vertice
                  and the Cartesian coordinates of each prism
    rmin: float - minimum value of radial distances
    rmax: float - maximum value of radial distances
    x0min: float - minimum value of x coordinate of origins
    x0max: float - maximum value of x coordinate of origins
    y0min: float - minimum value of y coordinate of origins
    y0max: float - maximum value of y coordinate of origins
    
    output
    
    mt: 1D array - parameters vector
    '''
    
    assert len(mt) == L*(M + 2), 'The size of m must be equal to L*(M + 2)'
    

    m = np.zeros_like(mt)
    
    for i in range(0, L*(M+2), M+2):
        m[i:M+i] = 0.001*rmin + 0.001*(rmax - rmin)/(1. + np.exp(-0.001*mt[i:M+i]))
        m[i+M] = 0.001*x0min + 0.001*(x0max - x0min)/(1. + np.exp(-0.001*mt[i+M]))
        m[i+M+1] = 0.001*y0min + 0.001*(y0max - y0min)/(1. + np.exp(-0.001*mt[i+M+1]))

    m *= 1000.
        
    return m


def trans_inv_parameter2(mt, M, L, mmax, mmin):
    '''
    Returns the parameters from the inverse transformation.
    
    input
    
    M: integer - number of vertices
    L: integer - number of prisms
    mt: 1D array - transformated parameters vector with 
                  radial distances of each vertice
                  and the Cartesian coordinates of each prism
    mmax: 1D array - maximum value of each parameter
                    (r1max,...,rMmax, x0max, y0max)
    mmin: 1D array - minimum value of each parameter
                    (r1min,...,rMmin, x0min, y0min)
    
    output
    
    mt: 1D array - parameters vector
    '''
    assert len(mmax) == L*(M + 2), 'The size of mmax must be equal to L*(M + 2)'
    assert len(mmin) == L*(M + 2), 'The size of mmin must be equal to L*(M + 2)'
    assert len(mt) == L*(M + 2), 'The size of m must be equal to L*(M + 2)'

    #m = 0.001*mmin + (0.001*(mmax - mmin))/(1. + np.exp(-0.001*mt))
    m = (mmin + (mmax - mmin)/(1. + np.exp(-mt)))

    return m
    

def gradient_data(xp, yp, zp, m, M, L, d, deltax, deltay, deltar, inc, dec):
    '''
    This function returns the gradient vector of the data
    from a model of polygonal prisms using finite difference.
    
    input
    
    xp: 1D array - x observation points
    yp: 1D array - y observation points
    zp: 1D array - z observation points
    m: list - each element is a list of [r, x0, y0, z1, z2, 'magnetization'],
              whrere r is an array with the radial distances of the vertices,
              x0 and y0 are the origin cartesian coordinates of each prism,
              z1 and z2 are the top and bottom of each prism and
              magnetization is physical property
    M: integer - number of vertices
    L: int - number of prisms
    d: 1D array - observed data vector
    deltax: float - increment in x coordinate in meters
    deltay: float - increment in y coordinate in meters
    deltar: float - increment in z coordinate in meters
    inc: float - inclination of the local-geomagnetic field
    dec: declination of the local-geomagnetic field
    
    output
    
    g: 2D array - gradient vector of the data
    '''
    assert len(m) == L, 'The size of m must be equal to L'
    assert xp.size == yp.size == zp.size, 'The number of points in x, y and z must be equal'
    
    model = pol2cart(m, M, L) # model with transformated parameters
    
    # predict data of transformated matrix
    dp = polyprism.tf(xp, yp, zp, model, inc, dec)
    
    # residual
    df = d - dp
    
    #Jacobian matrix
    G = fd_tf_sm_polyprism(xp, yp, zp, m, M, L, deltax, deltay, deltar, inc, dec)
    
    g = -2*np.dot(G.T, df)/xp.size
    
    return g

def Hessian_data(xp, yp, zp, m, M, L, deltax, deltay, deltar, inc, dec):
    '''
    This function returns the Hessian matrix of the data
    from a model of polygonal prisms using finite difference.
    
    input
    
    xp: array - x observation points
    yp: array - y observation points
    zp: array - z observation points
    m: list - each element is a list of [r, x0, y0, z1, z2, 'magnetization'],
              whrere r is an array with the radial distances of the vertices,
              x0 and y0 are the origin cartesian coordinates of each prism,
              z1 and z2 are the top and bottom of each prism and
              magnetization is physical property
    M: integer - number of vertices
    L: int - number of prisms
    deltax: float - increment in x coordinate in meters
    deltay: float - increment in y coordinate in meters
    deltar: float - increment in z coordinate in meters
    inc: float - inclination of the local-geomagnetic field
    dec: declination of the local-geomagnetic field
    
    output
    
    H: 2D array - Hessian matrix of the data
    '''
    assert len(m) == L, 'The size of m must be equal to L'
    assert xp.size == yp.size == zp.size, 'The number of points in x, y and z must be equal'
    
    #model = pol2cart(m, M, L) # model with transformated parameters
    
    #Jacobian matrix
    G = fd_tf_sm_polyprism(xp, yp, zp, m, M, L, deltax, deltay, deltar, inc, dec)
    
    H = 2*np.dot(G.T, G)/xp.size
    
    return H

def levmq(xp, yp, zp, lini, M, L, delta, lamb, mmin, mmax, mout, dobs, inc, dec):
    '''
    This function minimizes the objective function
    using the Levenberg-Marqudt algorithm.
    
    input
    
    xp: array - x observation points
    yp: array - y observation points
    zp: array - z observation points
    lini: list - each element is a list of [r, x0, y0, z1, z2, 'magnetization'],
              whrere r is an array with the radial distances of the vertices,
              x0 and y0 are the origin cartesian coordinates of each prism,
              z1 and z2 are the top and bottom of each prism and
              magnetization is physical property
    M: integer - number of vertices
    L: int - number of prisms
    delta: float - increment in x, y and z coordinate in meters
    lamb: float - Marquadt's parameter
    mmin: array - minimum values for each parameters (rmin, x0min, y0min)
    mmax: array - maximum values for each parameters (rmax, x0max, y0max)
    mout: array - parameters from the outcropping body (M+2)
    dobs: array - observated data
    inc: float - inclination of the local-geomagnetic field
    dec: declination of the local-geomagnetic field
    
    output
    
    d0: array - fitted data
    m0: array - estimated parameters
    l0: list - geometrical and physical parameters
    phi_list: list - solutions of objective funtion
    '''
    itmax = 50
    itmax_marq = 20
    epsilon = 0.0001
    lamb = 0.001
    dlamb = 5.
    z0 = lini[0][3]
    dz = lini[0][4] - lini[0][3]
    props = lini[0][5]
    l0 = deepcopy(lini)
    m0 = param_vec(l0, M, L) # inicial parameters vector
    model0 = pol2cart(l0, M, L) # list of classes of prisms
    d0 = polyprism.tf(xp, yp, zp, model0, inc, dec) # predict data
    dif = dobs - d0
    phi0 = np.sum(dif*dif)/xp.size
    phi_list = [phi0]
    alpha1 = 0.001
    alpha2 = 0.001
    alpha3 = 0.
    alpha4 = 0.
    alpha5 = alpha1
    alpha6 = 0.0001
    i,j = np.diag_indices(L*(M+2))

    for it in range(itmax):
        mt = trans_parameter2(m0, M, L, mmax, mmin)

        grad = gradient_data(xp, yp, zp, l0, M, L, dobs, delta, delta, delta, inc, dec)
        grad = gradient_phi_1(M, L, grad, alpha1)
        grad = gradient_phi_2(M, L, grad, alpha2)
        grad = gradient_phi_3(M, L, grad, mout, alpha3)
        grad = gradient_phi_4(M, L, grad, mout[-2:], alpha4)
        grad = gradient_phi_5(M, L, grad, alpha5)
        grad = gradient_phi_6(M, L, grad, alpha6)

        H = Hessian_data(xp, yp, zp, l0, M, L, delta, delta, delta, inc, dec)
        H = Hessian_phi_1(M, L, H, alpha1)
        H = Hessian_phi_2(M, L, H, alpha2)
        H = Hessian_phi_3(M, L, H, alpha3)
        H = Hessian_phi_4(M, L, H, alpha4)
        H = Hessian_phi_5(M, L, H, alpha5)
        H = Hessian_phi_6(M, L, H, alpha6)

        # Diagonal da matriz de positividade
        T = ((mmax - m0)*(m0 - mmin))/(mmax - mmin)

        for it_marq in range(itmax_marq): 

            S = H.copy()
            S[i,:] *= T
            S[i,j] += lamb

            delta_mt = np.linalg.solve(S, -grad)
            m_est = trans_inv_parameter2(mt + delta_mt, M, L, mmax, mmin)
            l_est = param2model(m_est, M, L, z0, dz, props)
            model_est = pol2cart(l_est, M, L)
            d_est = polyprism.tf(xp, yp, zp, model_est, inc, dec)
            res = dobs - d0
            phi = np.sum(res*res)/xp.size
            phi += (phi_1(M, L, m_est, alpha1) + phi_2(M, L, m_est, alpha2) + \
                    phi_3(M, L, m_est, mout, alpha3) + phi_4(M, L, m_est, mout[-2:], alpha4) + \
                    phi_5(M, L, m_est, alpha5) + phi_6(M, L, m_est, alpha6))
            dphi = phi - phi0

            if (dphi > 0.):
                lamb *= dlamb
            else:
                lamb /= dlamb

        if (abs((phi0 - phi)/phi0) > epsilon):
            d0 = d_est
            m0 = m_est
            l0 = l_est
            phi0 = phi
            phi_list.append(phi0)
            
        return d0, m0, l0, phi_list