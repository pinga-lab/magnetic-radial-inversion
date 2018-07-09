### Functions for magnetic problems with polygonal prisms

import numpy as np
import numpy.testing as npt
from fatiando import mesher, gridder, utils
from fatiando.gravmag import polyprism
from fatiando.mesher import PolygonalPrism
from fatiando.constants import CM, T2NT
from copy import deepcopy
from math import factorial

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
    assert x.size == y.size, 'x and y must have the same size'
    assert x.shape == y.shape, 'x, y and z must have the same shape'

    x = np.asanyarray(x)
    y = np.asanyarray(y)
    n = len(x)
    shift_up = np.arange(-n+1, 1)
    shift_down = np.arange(-1, n-1)
    area = (x * (y.take(shift_up) - y.take(shift_down))).sum() / 2.0
    return area

def pol2cart(l, M, L):
    '''
    This function transforms polar coordinates of the prisms
    into Cartesian coordinates and returns a list of polygonal
    prisms of the Fatiando a Terra.

    input

    l: list - each element is a list of [r, x0, y0, z1, z2, 'magnetization'],
              whrere r is an array with the radial distances of the vertices,
              x0 and y0 are the origin Cartesian coordinates of each prism,
              z1 and z2 are the top and bottom of each prism and
              magnetization is physical property
    M: int - number of vertices per prism
    L: int - number of prisms

    output

    mk: list - list of objects of the class
    fatiando.mesher.PolygonalPrism
    '''

    lk = []
    r = np.zeros(M)  # it contains radial distances of the vertices in polar coordinates
    verts = [] # it contains radial distrantances of the vertices in Cartesian coordinates

    assert len(l) == L, 'The size of m and the number of prisms must be equal'
    for lv in l:
        assert len(lv) == 6, 'Each element of l must have 6 elements'
        assert len(lv[0]) == M, 'All prisms must have M vertices'

    ang = 2*np.pi/M # angle between two vertices

    for lv in l:
        r = lv[0]
        verts = []
        for i in range(M):
            verts.append([r[i]*np.cos(i*ang) + lv[1], r[i]*np.sin(i*ang) + lv[2]])
        lk.append(PolygonalPrism(verts, lv[3], lv[4], lv[5]))

    return lk

def param_vec(l, M, L):
    '''
    This function receives the model of prisms and returns the vector of parameters

    input

    l: list - each element is a list of [r, x0, y0, z1, z2, 'magnetization'],
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
    lv = [] # list for the loop of asserts

    assert len(l) == L, 'The size of m and the number of prisms must be equal'

    for lv in l:
        assert len(lv) == 6, 'Each element of l must have 6 elements'
        assert len(lv[0]) == M, 'All prisms must have M vertices'
        assert lv[0][:M] > 0., 'All radius must be positives'

    for i in range(L):
        pv = np.hstack((pv, l[i][0], l[i][1:3]))
    pv = np.hstack((pv, l[0][4] - l[0][3]))

    return pv

def param2polyprism(m, M, L, z0, props):
    '''
    Returns a lis of objects of the class
    fatiando.mesher.PolygonalPrism

    input

    m: 1D array - parameter vector
    M: int - number of vertices
    L: int - number of prisms
    z0: float - top of the model
    props: dictionary - physical property

    output

    model: list - list of fatiando.mesher.PolygonalPrism
    '''
    P = L*(M + 2) + 1
    assert m.size == P, 'The size of m must be equal to L*(M + 2) + 1'
    #assert m[-1] > 0., 'The thickness dz must be a positive number'
    for i in range(P-1):
        assert m[i:i+M].all >= 0., 'The radial distances must be positives'

    r = np.zeros(M) # vector for radial distances
    model = [] # list of prisms

    k = 0.
    for i in range(0, P-1, M + 2):
        r = m[i:M+i]
        model.append([r, m[i+M], m[i+M+1], z0 + m[-1]*k, z0 + m[-1]*(k + 1.), props])
        k = k + 1.

    model = pol2cart(model, M, L)

    return model

### Functions for the derivatives with finite differences

def derivative_tf_x0(xp, yp, zp, m, M, delta, inc, dec):
    '''
    This function calculates the derivative for total field anomaly
    for x0 coordinate of a model of polygonal prisms using 
    finite difference.

    input

    xp, yp, zp: 1D array - observation points
    m: list - list of one fatiando.mesher.PolygonalPrism
    M: int - number of vertices per prism
    delta: float - increment for differentiation
    inc: float - inclination of the local-geomagnetic field
    dec: float - declination of the local-geomagnetic field

    output

    df: 1D array - derivative of x0 coordinate
    '''
    assert xp.size == yp.size == zp.size, 'The number of points in x, y and z must be equal'
    assert xp.shape == yp.shape == zp.shape, 'xp, yp and zp must have the same shape'
    assert m.x.size == m.y.size == M, 'The number of vertices must be M'
    assert delta > 0., 'delta must be a positive number'

    mp = deepcopy([m])  # m.x + delta
    mm = deepcopy([m])  # m.x - delta
    mp[0].x += delta
    mm[0].x -= delta

    df = polyprism.tf(xp, yp, zp, mp, inc, dec)
    df -= polyprism.tf(xp, yp, zp, mm, inc, dec)

    df /= (2.*delta)

    return df

def derivative_tf_y0(xp, yp, zp, m, M, delta, inc, dec):
    '''
    This function calculates the derivative for total field anomaly
    for y0 coordinate of a model of polygonal prisms using 
    finite difference.

    input

    xp, yp, zp: 1D array - observation points
    m: list - list of one fatiando.mesher.PolygonalPrism
    M: int - number of vertices per prism
    delta: float - increment for differentiation
    inc: float - inclination of the local-geomagnetic field
    dec: float - declination of the local-geomagnetic field

    output

    df: 1D array - derivative of x0 coordinate
    '''
    assert xp.size == yp.size == zp.size, 'The number of points in x, y and z must be equal'
    assert xp.shape == yp.shape == zp.shape, 'xp, yp and zp must have the same shape'
    assert m.x.size == m.y.size == M, 'The number of vertices must be M'
    assert delta > 0., 'delta must be a positive number'

    mp = deepcopy([m])  # m.y + delta
    mm = deepcopy([m])  # m.y - delta
    mp[0].y += delta
    mm[0].y -= delta

    df = polyprism.tf(xp, yp, zp, mp, inc, dec)
    df -= polyprism.tf(xp, yp, zp, mm, inc, dec)

    df /= (2.*delta)

    return df

def derivative_tf_radial(xp, yp, zp, m, M, nv, delta, inc, dec):
    '''
    This function calculates the derivative for total field anomaly
    for radial coordinate of a set of polygonal prisms using
    finite difference.

    input

    xp, yp, zp: 1D array - observation points
    m: list - list of a fatiando.mesher.PolygonalPrism
    M: int - number of vertices per prism
    nv: int - number of the vertice for the derivative
    delta: float - increment for differentiation
    inc: float - inclination of the local-geomagnetic field
    dec: float - declination of the local-geomagnetic field

    output

    df: 1D array - derivative of radial distance
    '''
    assert xp.size == yp.size == zp.size, 'The number of points in x, y and z must be equal'
    assert xp.shape == yp.shape == zp.shape, 'xp, yp and zp must have the same shape'
    assert m.x.size == m.y.size == M, 'The number of vertices must be M'
    assert nv < M, 'The vertice number must be smaller than the number of vertices (0 - M)'
    assert delta > 0., 'delta must be a positive number'

    m_fat = [] # list of objects of the class fatiando.mesher.PolygonalPrism
    verts = [] # vertices of new prism
    ang = 2.*np.pi/M # angle between two vertices

    if nv == M - 1:
        nvp = 0
    else:
        nvp = nv + 1

    deltax = delta*np.cos(nv*ang)
    deltay = delta*np.sin(nv*ang)

    verts.append([m.x[nv - 1], m.y[nv - 1]])
    verts.append([m.x[nv] + deltax, m.y[nv] + deltay])
    verts.append([m.x[nvp], m.y[nvp]])
    verts.append([m.x[nv] - deltax, m.y[nv] - deltay])

    m_fat = [PolygonalPrism(verts, m.z1, m.z2, m.props)]

    df = polyprism.tf(xp, yp, zp, m_fat, inc, dec)
    df /= (2.*delta)

    return df

def derivative_tf_radial2(xp, yp, zp, m, M, nv, delta, inc, dec):
    '''
    This function calculates the derivative for total field anomaly
    for radial coordinate of a set of polygonal prisms using
    finite difference.

    input

    xp, yp, zp: 1D array - observation points
    m: list - list of a fatiando.mesher.PolygonalPrism
    M: int - number of vertices per prism
    nv: int - number of the vertice for the derivative
    delta: float - increment for differentiation
    inc: float - inclination of the local-geomagnetic field
    dec: float - declination of the local-geomagnetic field

    output

    df: 1D array - derivative of radial distance
    '''
    assert xp.size == yp.size == zp.size, 'The number of points in x, y and z must be equal'
    assert xp.shape == yp.shape == zp.shape, 'xp, yp and zp must have the same shape'
    assert m.x.size == m.y.size == M, 'The number of vertices must be M'
    assert nv < M, 'The vertice number must be smaller than the number of vertices (0 - M)'
    assert delta > 0., 'delta must be a positive number'

    mp = deepcopy([m]) # list of objects of the class fatiando.mesher.PolygonalPrism
    mm = deepcopy([m])

    ang = 2.*np.pi/M # angle between two vertices

    deltax = delta*np.cos(nv*ang)
    deltay = delta*np.sin(nv*ang)

    mp[0].x[nv] += deltax
    mp[0].y[nv] += deltay

    mm[0].x[nv] -= deltax
    mm[0].y[nv] -= deltay

    df = polyprism.tf(xp, yp, zp, mp, inc, dec)
    df -= polyprism.tf(xp, yp, zp, mm, inc, dec)
    df /= (2.*delta)

    return df

def derivative_tf_dz(xp, yp, zp, m, L, delta, inc, dec):
    '''
    This function calculates the derivative for total field anomaly
    for thickness of a set of polygonal prisms using finite difference.

    input

    xp: array - x observation points
    yp: array - y observation points
    zp: array - z observation points
    m: list - list of L fatiando.mesher.PolygonalPrism
    L: int - number of prisms
    delta: float - increment for z coordinate in meters
    inc: float - inclination of the local-geomagnetic field
    dec: float - declination of the local-geomagnetic field

    output

    df: 1D array - derivative of dz
    '''
    assert xp.size == yp.size == zp.size, 'The number of points in x, y and z must be equal'
    assert xp.shape == yp.shape == zp.shape, 'xp, yp and zp must have the same shape'
    assert delta > 0., 'delta must be a positive number'
    
    mp = deepcopy(m)  # m.z + delta
    mm = deepcopy(m)  # m.z - delta
    mp[0].z2 += delta
    mm[0].z2 += delta
    for i in range(1, L, 1):
        mp[i].z1 += delta
        mp[i].z2 += delta
        mm[i].z1 -= delta
        mm[i].z2 -= delta

    df = polyprism.tf(xp, yp, zp, mp, inc, dec)
    df -= polyprism.tf(xp, yp, zp, mm, inc, dec)

    df /= (2.*delta)

    return df

def Jacobian_tf(xp, yp, zp, m, M, L, deltax, deltay, deltar, deltaz, inc, dec):
    '''
    Returns the sensitivity matrix for polygonal prisms using finite
    differences.

    input

    xp: array - x observation points
    yp: array - y observation points
    zp: array - z observation points
    m: list - list of fatiando.mesher.PolygonalPrism
    M: int - number of vertices per prism
    L: int - number of prisms
    deltax: float - increment for x coordinate in meters
    deltay: float - increment for y coordinate in meters
    deltar: float - increment for radial distances in meters
    deltaz: float - increment for z coordinate in meters
    inc: float - inclination of the local-geomagnetic field
    dec: declination of the local-geomagnetic field

    output

    G: 2D array - sensitivity matrix
    '''
    assert len(m) == L, 'The number of prisms must be L'
    for mv in m:
        assert mv.x.size == mv.y.size == M, 'The number of vertices must be M'
    assert xp.size == yp.size == zp.size, 'The number of points in x, y and z must be equal'
    assert xp.shape == yp.shape == zp.shape, 'xp, yp and zp must have the same shape'
    assert deltax > 0., 'deltax must be a positive number'
    assert deltay > 0., 'delaty must be a positive number'
    assert deltaz > 0., 'delatz must be a positive number'

    P = L*(M+2) + 1 # number of parameters per prism
    pp = M+2
    G = np.zeros((xp.size, P))
    G[:,-1] += derivative_tf_dz(xp, yp, zp, m, L, deltaz, inc, dec)

    for i, mv in enumerate(m):
        aux = i*pp
        G[:, aux + M] = derivative_tf_x0(xp, yp, zp, mv, M, deltax, inc, dec)
        G[:, aux + M + 1] = derivative_tf_y0(xp, yp, zp, mv, M, deltay, inc, dec)
        for j in range(M):
            G[:, aux + j] = derivative_tf_radial(xp, yp, zp, mv, M, j, deltar, inc, dec)
    
    return G

def derivative_amf_x0(xp, yp, zp, m, M, delta):
    '''
    This function calculates the derivative for amplitude of
    anomalous field for x0 coordinate of a set of polygonal
    prisms using finite difference.

    input

    xp, yp, zp: array - x observation points
    m: list - list of one fatiando.mesher.PolygonalPrism
    M: int - number of vertices per prism
    delta: float - increment for x coordinate in meters

    output

    df: 1D array - derivative
    '''
    assert xp.size == yp.size == zp.size, 'The number of points in x, y and z must be equal'
    assert xp.shape == yp.shape == zp.shape, 'xp, yp and zp must have the same shape'
    assert m.x.size == m.y.size == M, 'The number of vertices must be M'
    assert delta > 0., 'delta must be a positive number'

    mp = deepcopy([m])  # m.x + delta
    mm = deepcopy([m])  # m.x - delta
    mp[0].x += delta
    mm[0].x -= delta

    df = np.sqrt(polyprism.bx(xp, yp, zp, mp)**2. + \
                 polyprism.by(xp, yp, zp, mp)**2. + \
                 polyprism.bz(xp, yp, zp, mp)**2.)
    
    df -= np.sqrt(polyprism.bx(xp, yp, zp, mm)**2. + \
                  polyprism.by(xp, yp, zp, mm)**2. + \
                  polyprism.bz(xp, yp, zp, mm)**2.)
    
    df /= (2.*delta)

    return df

def derivative_amf_y0(xp, yp, zp, m, M, delta):
    '''
    This function calculates the derivative for amplitude of
    anomalous field of a set of polygonal prisms using 
    finite difference.

    input

    xp, yp, zp: array - observation points
    m: list - list of one fatiando.mesher.PolygonalPrism
    M: int - number of vertices per prism
    delta: float - increment for y coordinate in meters

    output

    df: 1D array - derivative
    '''
    assert xp.size == yp.size == zp.size, 'The number of points in x, y and z must be equal'
    assert xp.shape == yp.shape == zp.shape, 'xp, yp and zp must have the same shape'
    assert m.x.size == m.y.size == M, 'The number of vertices must be M'
    assert delta > 0., 'delta must be a positive number'

    mp = deepcopy([m])  # m.y + delta
    mm = deepcopy([m])  # m.y - delta
    mp[0].y += delta
    mm[0].y -= delta

    df = np.sqrt(polyprism.bx(xp, yp, zp, mp)**2. + \
                 polyprism.by(xp, yp, zp, mp)**2. + \
                 polyprism.bz(xp, yp, zp, mp)**2.)
    
    df -= np.sqrt(polyprism.bx(xp, yp, zp, mm)**2. + \
                  polyprism.by(xp, yp, zp, mm)**2. + \
                  polyprism.bz(xp, yp, zp, mm)**2.)

    df /= (2.*delta)

    return df

def derivative_amf_radial(xp, yp, zp, m, M, nv, delta):
    '''
    This function calculates the derivative for amplitude of
    anomalous field for radial coordinate of a set of 
    polygonal prisms using finite difference.

    input

    xp, yp, zp: array - observation points
    m: list - list of a fatiando.mesher.PolygonalPrism
    M: int - number of vertices per prism
    nv: int - number of the vertice for the derivative
    delta: float - increment for radial distance in meters

    output

    df: 1D array - derivative
    '''
    assert xp.size == yp.size == zp.size, 'The number of points in x, y and z must be equal'
    assert xp.shape == yp.shape == zp.shape, 'xp, yp and zp must have the same shape'
    assert m.x.size == m.y.size == M, 'The number of vertices must be M'
    assert delta > 0., 'delta must be a positive number'
    assert nv < M, 'The vertice number must be smaller than the number of vertices (0 - M)'

    m_fat = [] # list of objects of the class fatiando.mesher.PolygonalPrism
    verts = [] # vertices of new prism
    ang = 2.*np.pi/M # angle between two vertices

    if nv == M - 1:
        nvp = 0
    else:
        nvp = nv + 1

    deltax = delta*np.cos(nv*ang)
    deltay = delta*np.sin(nv*ang)

    verts.append([m.x[nv - 1], m.y[nv - 1]])
    verts.append([m.x[nv] + deltax, m.y[nv] + deltay])
    verts.append([m.x[nvp], m.y[nvp]])
    verts.append([m.x[nv] - deltax, m.y[nv] - deltay])

    m_fat = [PolygonalPrism(verts, m.z1, m.z2, m.props)]

    df = np.sqrt(polyprism.bx(xp, yp, zp, m_fat)**2. + \
                 polyprism.by(xp, yp, zp, m_fat)**2. + \
                 polyprism.bz(xp, yp, zp, m_fat)**2.)

    df /= (2.*delta)

    return df

def derivative_amf_dz(xp, yp, zp, m, L, delta):
    '''
    This function calculates the derivative for amplitude of
    anomalous field from a model of polygonal prisms using 
    finite difference.

    input

    xp: array - x observation points
    yp: array - y observation points
    zp: array - z observation points
    m: list - list of one fatiando.mesher.PolygonalPrism
    L: int - number of prisms
    delta: float - increment for z coordinate in meters

    output

    df: 1D array - derivative of dz
    '''
    assert xp.size == yp.size == zp.size, 'The number of points in x, y and z must be equal'
    assert xp.shape == yp.shape == zp.shape, 'xp, yp and zp must have the same shape'
    assert delta > 0., 'delta must be a positive number'

    mp = deepcopy(m)  # m.z + delta
    mm = deepcopy(m)  # m.z - delta
    mp[0].z2 += delta
    mm[0].z2 += delta
    for i in range(1, L, 1):
        mp[i].z1 += delta
        mp[i].z2 += delta
        mm[i].z1 -= delta
        mm[i].z2 -= delta

    df = np.sqrt(polyprism.bx(xp, yp, zp, mp)**2. + \
                 polyprism.by(xp, yp, zp, mp)**2. + \
                 polyprism.bz(xp, yp, zp, mp)**2.)
    
    df -= np.sqrt(polyprism.bx(xp, yp, zp, mm)**2. + \
                  polyprism.by(xp, yp, zp, mm)**2. + \
                  polyprism.bz(xp, yp, zp, mm)**2.)

    df /= (2.*delta)

    return df

def Jacobian_amf(xp, yp, zp, m, M, L, deltax, deltay, deltar, deltaz):
    '''
    Returns the sensitivity matrix for polygonal prisms using finite
    differences.

    input

    xp: array - x observation points
    yp: array - y observation points
    zp: array - z observation points
    m: list - list of fatiando.mesher.PolygonalPrism
    M: int - number of vertices per prism
    L: int - number of prisms
    deltax: float - increment for x coordinate in meters
    deltay: float - increment for y coordinate in meters
    deltar: float - increment for radial distances in meters
    deltaz: float - increment for z coordinate in meters

    output

    G: 2D array - sensitivity matrix
    '''
    assert len(m) == L, 'The number of prisms must be L'
    for mv in m:
        assert mv.x.size == mv.y.size == M, 'The number of vertices must be M'
    assert xp.size == yp.size == zp.size, 'The number of points in x, y and z must be equal'
    assert xp.shape == yp.shape == zp.shape, 'xp, yp and zp must have the same shape'
    assert deltax > 0., 'deltax must be a positive number'
    assert deltay > 0., 'delaty must be a positive number'
    assert deltaz > 0., 'delatz must be a positive number'

    P = L*(M+2) + 1 # number of parameters per prism
    pp = M+2
    G = np.zeros((xp.size, P))
    G[:,-1] += derivative_amf_dz(xp, yp, zp, m, L, deltaz)

    for i, mv in enumerate(m):
        aux = i*pp
        G[:, aux + M] = derivative_amf_x0(xp, yp, zp, mv, M, deltax)
        G[:, aux + M + 1] = derivative_amf_y0(xp, yp, zp, mv, M, deltay)
        for j in range(M):
            G[:, aux + j] = derivative_amf_radial(xp, yp, zp, mv, M, j, deltar)

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

    P = L*(M + 2) + 1

    assert H.shape == (P, P), 'The Hessians shape must be (P, P)'
    assert alpha >= 0., 'alpha must be greater or equal to 0'

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

    P = L*(M + 2) + 1

    assert H.shape == (P, P), 'The hessian shape must be (P, P)'
    assert alpha >= 0., 'alpha must be greater or equal to 0'

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

    P = L*(M + 2) + 1

    assert H.shape == (P, P), 'The hessian shape must be (P, P)'
    assert alpha >= 0., 'alpha must be greater or equal to 0'

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

    P = L*(M + 2) + 1

    assert H.shape == (P, P), 'The hessian shape must be (P, P)'
    assert alpha >= 0., 'alpha must be greater or equal to 0'

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

    P = L*(M + 2) + 1

    assert H.shape == (P, P), 'The hessian shape must be (P, P)'
    assert alpha >= 0., 'alpha must be greater or equal to 0'

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

    P = L*(M + 2) + 1

    assert H.shape == (P, P), 'The hessian shape must be (P, P)'
    assert alpha >= 0., 'alpha must be greater or equal to 0'

    # extracting the non-zero diagonals
    d0 = diags_phi_6(M, L, alpha)

    i, j = np.diag_indices_from(H) # indices of the diagonal elements

    H[i,j] += d0

    return H

def Hessian_phi_7(M, L, H, alpha):
    '''
    Returns the hessian matrix for Tikhonov's zero order
    for dz parameter.

    input

    M: integer - number of vertices
    L: integer - number of prisms
    H: 2D array - hessian matrix
    alpha: float - weight

    output

    H: 2D array - hessian matrix plus phi_7 constraint
    '''

    P = L*(M + 2) + 1

    assert H.shape == (P, P), 'The hessian shape must be (P, P)'
    assert alpha >= 0., 'alpha must be greater or equal to 0'

    H[-1,-1] += alpha

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

    P = L*(M + 2) + 1

    assert m.size == P, 'The size of parameter vector must be equal to P'
    assert alpha >= 0., 'alpha must be greater or equal to 0'

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

    P = L*(M + 2) + 1

    assert m.size == P, 'The size of parameter vector must be equal to P'
    assert alpha >= 0., 'alpha must be greater or equal to 0'

    m2 = m.copy() # the new vector m2 = gradient input + gradient of phi2

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

    P = L*(M + 2) + 1

    assert m.size == P, 'The size of parameter vector must be equal to P'
    assert m0.size == M + 2, 'The size of parameter vector must be equal to M + 2'
    assert alpha >= 0., 'alpha must be greater or equal to 0'
    
    m3 = np.copy(m) # the new vector m3 = gradient input + gradient of phi3
    
    # calculating the product between the diagonals and the slices of m
    m3[:M+2] += (m[:M+2] - m0)*alpha

    return m3

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

    P = L*(M + 2) + 1

    assert m.size == P, 'The size of parameter vector must be equal to P'
    assert m0.size == 2, 'The size of parameter vector must be equal to 2'
    assert alpha >= 0., 'alpha must be greater or equal to 0'

    m4 = np.copy(m) # the new vector m4 = gradient input + gradient of phi4

    # calculating the product between the diagonals and the slices of m
    m4[M:M+2] += (m[M:M+2] - m0)*alpha

    return m4

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

    P = L*(M + 2) + 1

    assert m.size == P, 'The size of parameter vector must be equal to P'
    assert alpha >= 0., 'alpha must be greater or equal to 0'

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

    P = L*(M + 2) + 1

    assert m.size == P, 'The size of parameter vector must be equal to P'
    assert alpha >= 0., 'alpha must be greater or equal to 0'

    m6 = m.copy() # the new vector m1 = gradient input + gradient of phi6

    # extracting the non-zero diagonals
    d0 = diags_phi_6(M, L, alpha)

    # calculating the product between the diagonals and the slices of m
    m6 += m*d0

    return m6

def gradient_phi_7(M, L, m, alpha):
    '''
    Returns the gradient vector for Tikhonov's zero order
    for dz parameter.

    input

    M: integer - number of vertices
    L: integer - number of prisms
    m: 1D array - gradient of parameter vector
    alpha: float - weight

    output

    m: 1D array - gradient vector plus phi_7 constraint
    '''

    P = L*(M + 2) + 1

    assert m.size == P, 'The size of parameter vector must be equal to P'
    assert alpha >= 0., 'alpha must be greater or equal to 0'

    m7 = m.copy() # the new vector m1 = gradient input + gradient of phi7

    # calculating the product between the diagonals and the slices of m
    m7[-1] += m[-1]*alpha

    return m7

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

    P = L*(M + 2) + 1

    assert m.size == P, 'The size of parameter vector must be equal to P'
    assert alpha >= 0., 'alpha must be greater or equal to 0'

    m1 = m.copy()

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

    P = L*(M + 2) + 1

    assert m.size == P, 'The size of parameter vector must be equal to P'
    assert alpha >= 0., 'alpha must be greater or equal to 0'

    # extracting the non-zero diagonals
    d0, d1 = diags_phi_2(M, L, alpha)

    m2 = m.copy()

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

    P = L*(M + 2) + 1

    assert m.size == P, 'The size of parameter vector must be equal to P'
    assert m0.size == M + 2, 'The size of parameter vector must be equal to M + 2'
    assert alpha >= 0., 'alpha must be greater or equal to 0'

    m3 = np.zeros(M+2)

    # calculating the product between the diagonals and the slices of m
    m3 = (m[:M+2] - m0)*alpha

    phi_3 = np.dot(m3, m3)

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

    P = L*(M + 2)  + 1

    assert m.size == P, 'The size of parameter vector must be equal to P'
    assert m0.size == 2, 'The size of parameter vector must be equal to 2'
    assert alpha >= 0., 'alpha must be greater or equal to 0'

    m4 = np.zeros(2)

    # calculating the product between the diagonals and the slices of m
    m4 = (m[M:M+2] - m0)*alpha

    phi_4 = np.dot(m4, m4)

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

    P = L*(M + 2) + 1

    assert m.size == P, 'The size of parameter vector must be equal to P'
    assert alpha >= 0., 'alpha must be greater or equal to 0'

    m5 = m.copy()

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

    P = L*(M + 2) + 1

    assert m.size == P, 'The size of parameter vector must be equal to P'
    assert alpha >= 0., 'alpha must be greater or equal to 0'

    m6 = m.copy()

    # extracting the non-zero diagonals
    d0 = diags_phi_6(M, L, alpha)

    # calculating the product between the diagonals and the slices of m
    m6 = m*d0

    phi_6 = np.dot(m6, m)

    return phi_6

def phi_7(M, L, m, alpha):
    '''
    Returns the value for the phi7 constraint.

    input

    M: integer - number of vertices
    L: integer - number of prisms
    m: 1D array - parameter vector
    alpha: float - weight

    output

    phi_7: float - value of phi_7 constraint
    '''

    P = L*(M + 2) + 1

    assert m.size == P, 'The size of parameter vector must be equal to P'
    assert alpha >= 0., 'alpha must be greater or equal to 0'

    m7 = m.copy()

    phi_7 = m7[-1]*m7[-1]*alpha

    return phi_7

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
    assert alpha >= 0., 'alpha must be greater or equal to 0'
    
    P = L*(M + 2)

    # building the diagonals
    d0 = np.zeros(M+2)
    d0[:M] = 2.*alpha
    d0 = np.resize(d0, P)
    d0 = np.hstack((d0, 0.))

    d1 = np.zeros(M+2)
    d1[:M-1] = - alpha
    d1 = np.resize(d1, P-1)
    d1 = np.hstack((d1, 0.))

    dM = np.zeros(M+2)
    dM[0] = - alpha
    dM = np.resize(dM, P-M+1)
    dM = np.hstack((dM, 0.))

    return d0, d1, dM

def norm_regul_param(M, L, th, m0, a1, a2, a3, a4, a5, a6):
    '''
    Returns the normalized regularization parameters of each phi.

    input

    M: integer - number of vertices
    L: integer - number of prisms
    th: float - trace of the Hessian of initial model
    a1: float - weight of phi1
    a2: float - weight of phi2
    a3: float - weight of phi3
    a4: float - weight of phi4
    a5: float - weight of phi5
    a6: float - weight of phi6

    output

    alpha1: float - phi1 normalized regularization parameter
    alpha2: float - phi2 normalized regularization parameter
    alpha3: float - phi3 normalized regularization parameter
    alpha4: float - phi4 normalized regularization parameter
    alpha5: float - phi5 normalized regularization parameter
    alpha6: float - phi6 normalized regularization parameter
    '''

    # phi1
    alpha1 = a1*(th/(2.*L*M))

    # phi2
    if L <= 2:
        alpha2 = a2*(th/(L*M))
    else:
        alpha2 = a2*(th/(2.*(L-1)*M))

    # phi3
    m3 = np.ones(M+2)
    m3 = (m3 - m0)
    alpha3 = a3*(th/np.sum(m3))

    # phi4
    m4 = np.ones(2)
    m4 = (m4 - m0[M:M+2])
    alpha4 = a4*(th/np.sum(m4))

    # phi5
    if L == 2:
        alpha5 = a5*(th/(2.*L))
    else:
        alpha5 = a5*(th/(2.*(L-1)))

    # phi6
    alpha6 = a6*(th/(L*M))

    alpha1 = th*a1
    alpha2 = th*a2
    alpha3 = th*a3
    alpha4 = th*a4
    alpha5 = th*a5
    alpha6 = th*a6

    return alpha1, alpha2, alpha3, alpha4, alpha5, alpha6

def diags_phi_2(M, L, alpha):
    '''
    Returns the non-zero diagonals of hessian matrix for
    the smoothness constraint on adjacent radial distances
    in the adjacent prisms.

    input

    M: integer - number of vertices
    L: integer - number of prisms

    output

    d0, d1: 1D array - diagonals from phi_2 hessian
    '''
    assert L >= 2, 'The number of prisms must be greater than 1'
    assert alpha >= 0., 'alpha must be greater or equal to 0'

    P = L*(M + 2)

    # building the diagonals

    d0 = np.zeros(M+2)

    if L <= 2:
        d0[:M] = alpha
        d0 = np.resize(d0, P)
        d0 = np.hstack((d0, 0.))
    else:
        d0[:M] = 2.*alpha
        d0 = np.resize(d0, P)
        d0[:M] -= alpha
        d0[-M-2:-2] -= alpha
        d0 = np.hstack((d0, 0.))

    d1 = np.zeros(M+2)
    d1[:M] = - alpha
    d1 = np.resize(d1, P-M-2)
    d1 = np.hstack((d1, 0.))

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
    assert alpha >= 0., 'alpha must be greater or equal to 0'

    P = L*(M + 2)

    # building the diagonals
    d0 = np.zeros(M+2)

    if L == 2:
        d0[M:M+2] = alpha
        d0 = np.resize(d0, P)
        d0 = np.hstack((d0, 0.))
    else:
        d0[M:M+2] = 2*alpha
        d0 = np.resize(d0, P)
        d0[M:M+2] -= alpha
        d0[-2:] -= alpha
        d0 = np.hstack((d0, 0.))

    d1 = np.zeros(M+2)
    d1[M:M+2] -= alpha
    d1 = np.resize(d1, P-M-2)
    d1 = np.hstack((d1, 0.))

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
    assert alpha >= 0., 'alpha must be greater or equal to 0'

    P = L*(M + 2)

    # building the diagonal
    d0 = np.zeros(M+2)
    d0[:M] += alpha
    d0 = np.resize(d0, P)
    d0 = np.hstack((d0, 0.))

    return d0

# Functions for inverse problem

def build_range_param(M, L, rmin, rmax, x0min, x0max, y0min, y0max, dzmin, dzmax):
    '''
    Returns vectors of maximum and minimum values of
    parameters
    input
    rmin: float - minimum value of radial distances
    rmax: float - maximum value of radial distances
    x0min: float - minimum value of x Cartesian coordinate of the origins
    x0max: float - maximum value of x Cartesian coordinate of the origins
    y0max: float - minimum value of y Cartesian coordinate of the origins
    y0min: float - maximum value of y Cartesian coordinate of the origins
    dzmin: float - minimum value of thickness dz of each prism
    dzmax: float - maximum value of thickness dz of each prism
    output
    mmin: 1D array - vector of minimum values of parameters
    mmax: 1D array - vector of maximum values of parameters
    '''
    assert rmin >= 0., 'The minimum value of radial distances must be positive'
    assert rmax >= 0., 'The maximum value of radial distances must be positive'
    assert dzmin >= 0., 'The maximum value of dzmin must be positive'
    assert dzmax >= 0., 'The maximum value of dzmax must be positive'

    P = L*(M+2)
    mmax = np.zeros(M+2)
    mmin = np.zeros(M+2)

    mmax[:M] = rmax
    mmax[M] = x0max
    mmax[M+1] = y0max
    mmin[:M] = rmin
    mmin[M] = x0min
    mmin[M+1] = y0min

    mmax = np.resize(mmax, P)
    mmax = np.hstack((mmax, dzmax))
    mmin = np.resize(mmin, P)
    mmin = np.hstack((mmin, dzmin))

    return mmin, mmax

def log_barrier(m, M, L, mmax, mmin):
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
    P = L*(M+2) + 1
    assert mmax.size == mmin.size == m.size == P, 'The size of mmax, mmin, and m must be equal to P'
    assert mmax.shape == mmin.shape == m.shape == (P,), 'The shape of mmax, mmin, and m must be equal to (P,)'
    assert np.alltrue(m <= mmax), 'mmax must be greater than m'
    assert np.alltrue(m >= mmin), 'm must be greater than mmin'

    mt = - np.log((mmax - m)/(m - mmin + 1e-2))
    
    return mt

def inv_log_barrier(mt, M, L, mmax, mmin):
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

    p: 1D array - parameters vector
    '''

    P = L*(M+2) + 1
    assert mmax.size == mmin.size == mt.size == P, 'The size of mmax, mmin, and mt must be equal to P'
    assert mmax.shape == mmin.shape == mt.shape == (P,), 'The shape of mmax, mmin, and m must be equal to (P,)'
    
    i_overflow = np.argwhere(mt <= -710.)
    mt[i_overflow] = -700.
    
    m = mmin + (mmax - mmin)/(1. + np.exp(-mt))
    
    i_max = np.argwhere(m >= mmax)
    i_min = np.argwhere(m <= mmin)
    m[i_max] = mmax[i_max] - 1e-1
    m[i_min] = mmin[i_min] + 1e-1
    
    return m

def levmarq_tf(xp, yp, zp, m0, M, L, delta, maxit, maxsteps, lamb, dlamb, tol, mmin, mmax, m_out, dobs, inc, dec, props, alpha, z0, dz):
    '''
    This function minimizes the goal function of a set of polygonal prism
    for total-field-anomaly using the Levenberg-Marqudt algorithm.

    input

    xp, yp, zp: 1D array - observation points
    m0: 1D array - initial parameters vector
    M: integer - number of vertices
    L: int - number of prisms
    delta: 1D vector - (deltax, deltay, deltar, deltaz) increments for x, y, r and z coordinate in meters
    maxit: int - number of iterations
    maxsteps: int - number of steps
    lamb: float - Marquadt's parameter
    dlamb: float - variation of Marquadt's parameter
    tol: float - convergence criterion
    mmin: array - minimum values for each parameters (rmin, x0min, y0min)
    mmax: array - maximum values for each parameters (rmax, x0max, y0max)
    m_out: array - parameters from the outcropping body (M+2)
    dobs: array - observed data
    inc, dec: float - inclination and declination of the local-geomagnetic field
    props: dictionary - direction of magnetization
    alpha: 1D vector - (a1, a2, a3, a4 , a5, a6, a7) regularization parameters
    z0: float - the top of the source
    dz: float - thickness of the prisms
    output

    d0: array - fitted data
    m0: array - estimated parameters
    model0: list - objects of fatiando.mesher.polyprisms
    phi_list: list - solutions of objective funtion
    '''
    P = L*(M + 2) + 1
    assert xp.size == yp.size == zp.size, 'The number of points in x, y and z must be equal'
    assert xp.shape == yp.shape == zp.shape, 'xp, yp and zp must have the same shape'
    assert m0.size == P, 'The size of m0 must be equal to P'
    assert m0.shape == (P,), 'The shape of m0 must be equal to (P,)'
    assert np.alltrue > (alpha.all >= 0.), 'The regularization parameters must be positive or zero'
    assert dz > 0., 'dz must be a positive number'
    assert lamb > 0., 'lamb must be a positive number'
    assert dlamb > 0., 'dlamb must be a positive number'
    assert tol > 0., 'tol must be a positive number'

    model0 = param2polyprism(m0, M, L, z0, props) # list of classes of prisms
    d0 = polyprism.tf(xp, yp, zp, model0, inc, dec) # predict data
    res0 = dobs - d0
    phi0 = np.sum(res0*res0)/xp.size
    phi_list = [phi0]
    N = xp.size

    for it in range(maxit):
        mt = log_barrier(m0, M, L, mmax, mmin)

        # Jacobian matrix
        G = Jacobian_tf(xp, yp, zp, model0, M, L, delta[0], delta[1], delta[2], delta[3], inc, dec)

        # Hessian matrix
        H = 2.*np.dot(G.T, G)/N
        th = np.trace(H)/P

        # weighting the regularization parameters
        mu = alpha*th

        H = Hessian_phi_1(M, L, H, mu[0])
        H = Hessian_phi_2(M, L, H, mu[1])
        H = Hessian_phi_3(M, L, H, mu[2])
        H = Hessian_phi_4(M, L, H, mu[3])
        H = Hessian_phi_5(M, L, H, mu[4])
        H = Hessian_phi_6(M, L, H, mu[5])
        H = Hessian_phi_7(M, L, H, mu[6])

        # gradient vector
        grad = -2.*np.dot(G.T, res0)/N

        grad = gradient_phi_1(M, L, grad, mu[0])
        grad = gradient_phi_2(M, L, grad, mu[1])
        grad = gradient_phi_3(M, L, grad, m_out, mu[2])
        grad = gradient_phi_4(M, L, grad, m_out[-2:], mu[3])
        grad = gradient_phi_5(M, L, grad, mu[4])
        grad = gradient_phi_6(M, L, grad, mu[5])
        grad = gradient_phi_7(M, L, grad, mu[6])

        # positivity constraint
        H *= ((mmax - m0 + 1e-10)*(m0 - mmin + 1e-10))/(mmax - mmin)

        # Hessian normalization
        D = 1./np.sqrt(np.diag(H))
        
        for it_marq in range(maxsteps):
            
            delta_mt = (D*(np.linalg.solve((D*(H*D).T).T + lamb*np.identity(mt.size), -D*grad)).T).T
            m_est = inv_log_barrier(mt + delta_mt, M, L, mmax, mmin)
            model_est = param2polyprism(m_est, M, L, z0, props)
            d_fit = polyprism.tf(xp, yp, zp, model_est, inc, dec)
            res = dobs - d_fit
            phi = np.sum(res*res)/N
            phi += phi_1(M, L, m_est, mu[0]) + \
                    phi_2(M, L, m_est, mu[1]) + \
                    phi_3(M, L, m_est, m_out, mu[2]) + \
                    phi_4(M, L, m_est, m_out[-2:], mu[3]) + \
                    phi_5(M, L, m_est, mu[4]) + \
                    phi_6(M, L, m_est, mu[5]) + \
                    phi_7(M, L, m_est, mu[6])

            dphi = phi - phi0

            print 'it: %2d   it_marq: %2d   lambda: %.e   misfit: %.4e' % (it, it_marq, lamb, phi)

            if (dphi > 0.):
                lamb *= dlamb
            else:
                if lamb/dlamb < 1e-15:
                    lamb = 1e-15
                else:
                    lamb /= dlamb
                break

        if (abs(dphi/phi0) < tol):
            break
        else:
            d0 = d_fit.copy()
            m0 = m_est.copy()
            model0 = model_est
            res0 = res.copy()
            phi0 = phi
            phi_list.append(phi0)

    return d0, m0, model0, phi_list

def levmarq_amf(xp, yp, zp, m0, M, L, delta, maxit, maxsteps, lamb, dlamb, tol, mmin, mmax, m_out, dobs, props, alpha, z0, dz):
    '''
    This function minimizes the goal function of a set of polygonal prism
    for anomalous magnetic field amplitude using the Levenberg-Marqudt algorithm.

    input

    xp, yp, zp: 1D array - observation points
    m0: 1D array - initial parameters vector
    M: integer - number of vertices
    L: int - number of prisms
    delta: 1D vector - (deltax, deltay, deltar, deltaz) increments for x, y, r and z coordinate in meters
    maxit: int - number of iterations
    maxsteps: int - number of steps
    lamb: float - Marquadt's parameter
    dlamb: float - variation of Marquadt's parameter
    tol: float - convergence criterion
    mmin: array - minimum values for each parameters (rmin, x0min, y0min)
    mmax: array - maximum values for each parameters (rmax, x0max, y0max)
    m_out: array - parameters from the outcropping body (M+2)
    dobs: array - observed data
    props: dictionary - direction of magnetization
    alpha: 1D vector - (a1, a2, a3, a4 , a5, a6, a7) regularization parameters
    z0: float - the top of the source
    dz: float - thickness of the prisms
    output

    d0: array - fitted data
    m0: array - estimated parameters
    model0: list - objects of fatiando.mesher.polyprisms
    phi_list: list - solutions of objective funtion
    '''
    P = L*(M + 2) + 1
    assert xp.size == yp.size == zp.size, 'The number of points in x, y and z must be equal'
    assert xp.shape == yp.shape == zp.shape, 'xp, yp and zp must have the same shape'
    assert m0.size == P, 'The size of m0 must be equal to P'
    assert m0.shape == (P,), 'The shape of m0 must be equal to (P,)'
    assert np.alltrue > (alpha.all >= 0.), 'The regularization parameters must be positive or zero'
    assert dz > 0., 'dz must be a positive number'
    assert lamb > 0., 'lamb must be a positive number'
    assert dlamb > 0., 'dlamb must be a positive number'
    assert tol > 0., 'tol must be a positive number'

    model0 = param2polyprism(m0, M, L, z0, props) # list of classes of prisms
    d0 = np.sqrt(polyprism.bx(xp, yp, zp, model0)**2. + \
                 polyprism.by(xp, yp, zp, model0)**2. + \
                 polyprism.bz(xp, yp, zp, model0)**2.)
    res0 = dobs - d0
    phi0 = np.sum(res0*res0)/xp.size
    phi_list = [phi0]
    N = xp.size

    for it in range(maxit):
        mt = log_barrier(m0, M, L, mmax, mmin)

        # Jacobian matrix
        G = Jacobian_amf(xp, yp, zp, model0, M, L, delta[0], delta[1], delta[2], delta[3])

        # Hessian matrix
        H = 2.*np.dot(G.T, G)/N
        th = np.trace(H)/P

        # weighting the regularization parameters
        mu = alpha*th

        H = Hessian_phi_1(M, L, H, mu[0])
        H = Hessian_phi_2(M, L, H, mu[1])
        H = Hessian_phi_3(M, L, H, mu[2])
        H = Hessian_phi_4(M, L, H, mu[3])
        H = Hessian_phi_5(M, L, H, mu[4])
        H = Hessian_phi_6(M, L, H, mu[5])
        H = Hessian_phi_7(M, L, H, mu[6])

        # gradient vector
        grad = -2.*np.dot(G.T, res0)/N

        grad = gradient_phi_1(M, L, grad, mu[0])
        grad = gradient_phi_2(M, L, grad, mu[1])
        grad = gradient_phi_3(M, L, grad, m_out, mu[2])
        grad = gradient_phi_4(M, L, grad, m_out[-2:], mu[3])
        grad = gradient_phi_5(M, L, grad, mu[4])
        grad = gradient_phi_6(M, L, grad, mu[5])
        grad = gradient_phi_7(M, L, grad, mu[6])

        # positivity constraint
        H *= ((mmax - m0 + 1e-10)*(m0 - mmin + 1e-10))/(mmax - mmin)

        # Hessian normalization
        D = 1./np.sqrt(np.diag(H))
        
        for it_marq in range(maxsteps):
            
            delta_mt = (D*(np.linalg.solve((D*(H*D).T).T + lamb*np.identity(mt.size), -D*grad)).T).T
            m_est = inv_log_barrier(mt + delta_mt, M, L, mmax, mmin)
            model_est = param2polyprism(m_est, M, L, z0, props)
            d_fit = np.sqrt(polyprism.bx(xp, yp, zp, model_est)**2. + \
                            polyprism.by(xp, yp, zp, model_est)**2. + \
                            polyprism.bz(xp, yp, zp, model_est)**2.)
            res = dobs - d_fit
            phi = np.sum(res*res)/N
            phi += phi_1(M, L, m_est, mu[0]) + \
                    phi_2(M, L, m_est, mu[1]) + \
                    phi_3(M, L, m_est, m_out, mu[2]) + \
                    phi_4(M, L, m_est, m_out[-2:], mu[3]) + \
                    phi_5(M, L, m_est, mu[4]) + \
                    phi_6(M, L, m_est, mu[5]) + \
                    phi_7(M, L, m_est, mu[6])

            dphi = phi - phi0

            #print 'it: %2d   it_marq: %2d   lambda: %.e   misfit: %.4e' % (it, it_marq, lamb, phi)

            if (dphi > 0.):
                lamb *= dlamb
            else:
                if lamb/dlamb < 1e-15:
                    lamb = 1e-15
                else:
                    lamb /= dlamb
                break

        if (abs(dphi/phi0) < tol):
            break
        else:
            d0 = d_fit.copy()
            m0 = m_est.copy()
            model0 = model_est
            res0 = res.copy()
            phi0 = phi
            phi_list.append(phi0)

    return d0, m0, model0, phi_list