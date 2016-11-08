### Functions for magnetic problems with polygonal prisms

import numpy as np
from fatiando import mesher, gridder, utils
from fatiando.gravmag import polyprism
from fatiando.mesher import PolygonalPrism
from fatiando.vis import mpl
from fatiando.constants import CM, T2NT

### Functions for the foward problem using fatiando

def pol2cart(m, Np, Nv):
    '''
    This function transforms polar coordinates of the prisms
    into Cartesian coordinates and return a list of polygonal
    prisms of the Fatiando a Terra.
    
    input
    
    m: list - each element is a list [r, x0, y0, z1, z2, 'magnetization'],
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
    delta: float - variation for finite difference in meters
    inc: float - inclination
    dec: declination
    
    output
    
    df: array - derivative
    '''
    assert len(m[0]) + len(m[1:]) == Nv + 5, 'The number of parameter must be Nv + 2'
    
    mp = []  # m + delta
    mm = []  # m - delta
    mp_fat = [] # list of objects of the class fatiando.mesher.PolygonalPrism
    mm_fat = [] # list of objects of the class fatiando.mesher.PolygonalPrism    
    df = np.zeros(xp.size) # derivative
    
    mp = [[m[0], m[1] + delta, m[2], m[3], m[4], m[5]]]
    mm = [[m[0], m[1] - delta, m[2], m[3], m[4], m[5]]]
    
    mp_fat = pol2cart(mp, 1, Nv)
    mm_fat = pol2cart(mm, 1, Nv)
    
    df = (polyprism.tf(xp, yp, zp, mp_fat, inc, dec) - polyprism.tf(xp, yp, zp, mm_fat, inc, dec))/2.*delta
    
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
    delta: float - variation for finite difference in meters
    inc: float - inclination
    dec: declination
    
    output
    
    df: array - derivative
    '''
    assert len(m[0]) + len(m[1:]) == Nv + 5, 'The number of parameter must be Nv + 2'
    
    mp = []  # m + delta
    mm = []  # m - delta
    mp_fat = [] # list of objects of the class fatiando.mesher.PolygonalPrism
    mm_fat = [] # list of objects of the class fatiando.mesher.PolygonalPrism    
    df = np.zeros(xp.size) # derivative
    
    mp = [[m[0], m[1], m[2] + delta, m[3], m[4], m[5]]]
    mm = [[m[0], m[1], m[2] - delta, m[3], m[4], m[5]]]
    
    mp_fat = pol2cart(mp, 1, Nv)
    mm_fat = pol2cart(mm, 1, Nv)
    
    df = (polyprism.tf(xp, yp, zp, mp_fat, inc, dec) - polyprism.tf(xp, yp, zp, mm_fat, inc, dec))/2.*delta
    
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
    delta: float - variation for finite difference in meters
    inc: float - inclination
    dec: declination
    
    output
    
    df: array - derivative
    '''
    assert len(m[0]) + len(m[1:]) == Nv + 5, 'The number of parameter must be Nv + 2'
    assert nv <= Nv, 'The vertice number must be minor or equal to the number of vertices'
    
    m_fat = [] # list of objects of the class fatiando.mesher.PolygonalPrism
    df = np.zeros(xp.size) # derivative
    verts = [] # vertices of new prism
    ang = 2.*np.pi/Nv # angle between two vertices
    
    if nv == Nv:
        nvp = -1
    else:
        nvp = nv
    
    verts.append([m[0][nv - 1]*np.cos(ang), m[0][nv - 1]*np.sin(ang)])
    verts.append([(m[0][nv] + delta)*np.cos(ang), (m[0][nv] + delta)*np.sin(ang)])
    verts.append([m[0][nvp]*np.cos(ang), m[0][nvp]*np.sin(ang)])
    verts.append([(m[0][nv] - delta)*np.cos(ang), (m[0][nv] - delta)*np.sin(ang)])

    m_fat = [PolygonalPrism(verts, m[3], m[4], m[5])]
    
    df = polyprism.tf(xp, yp, zp, m_fat, inc, dec)
    
    return df