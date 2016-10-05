### Functions for magnetic problems with polygonal prisms

import numpy as np
from fatiando import mesher, gridder, utils
from fatiando.gravmag import polyprism
from fatiando.mesher import PolygonalPrism
from fatiando.vis import mpl
from fatiando.constants import CM, T2NT

### Functions for the foward problem using fatiando

def pol2cart(m, x0, y0, Np, Nv):
    '''
    This function transforms polar coordinates of the prisms
    into Cartesian coordinates and return a list of polygonal
    prisms of the Fatiando a Terra.
    
    input
    
    m: list - each element is a list [verts,z0,R,magnetization] 
    containing the vertices in polar coordinates
    Np: int - number of prisms
    Nv: int - number of vertices per prism
    
    output
    
    mk: list - list of objects of the class
    fatiando.mesher.PolygonalPrism
    '''
    
    mk = []
    verts = []  # it contains vertices in polar coordinates
    vertsk = [] # it contains vertices in Cartesian coordinates
 
    assert len(m) == Np, 'The size of m and the number of prisms must be equal'
    
    for mv in m:
        assert len(mv[0]) == Nv, 'All prisms must have Nv vertices'
        
    assert len(m) = x0.size ...

    for i, mv in enumerate(m):
        verts = mv[0]
        for v in verts:
            vertsk.append([v[0]*np.cos(v[1]) + x0[i], v[0]*np.sin(v[1]) + y0[i]])
        mk.append(PolygonalPrism(vertsk, mv[1], mv[2], mv[3]))
        
        
    return mk