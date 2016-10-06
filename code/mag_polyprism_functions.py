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
        
    #assert len(m) == x0.size == y0.size, 'Each prism must have an origin coordinates'
    
    ang = 2*np.pi/Nv # angle between two vertices

    for mv in m:
        r = mv[0]
        for i in range(Nv):
            verts.append([r[i]*np.cos(i*ang) + mv[1], r[i]*np.sin(i*ang) + mv[2]])
        mk.append(PolygonalPrism(verts, mv[3], mv[4], mv[5]))
        
    return mk