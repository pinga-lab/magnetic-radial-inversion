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
    This function transforms polar coordinates from the prisms
    into cartesian coordinates and return.
    
    input
    
    m: [verts,z0,R,magnetization] - verts in polar coordinates
    Np: number of prisms
    Nv: number of vertices per prism
    
    output
    
    mk: list - list of classes of polygonal prisms from fatiando
    '''
    
    mk = []
    verts = []
    vertsk = []
 
    assert len(m) == Np, 'The size of m and the number of prisms must be equal'

    for mv in m:
        verts = mv[0]
        for v in verts:
            vertsk.append([v[0]*np.cos(v[1]), v[0]*np.sin(v[1])])
        mk.append(PolygonalPrism(vertsk, mv[1], mv[2], mv[3]))
        
        
    return mk