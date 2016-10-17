### Test functions for magnetic problems with polygonal prisms

import numpy as np
from fatiando import mesher, gridder, utils
from fatiando.gravmag import polyprism
from fatiando.mesher import PolygonalPrism
from fatiando.gravmag import prism
from fatiando.mesher import Prism
from fatiando.vis import mpl
from fatiando.constants import CM, T2NT
import mag_polyprism_functions as mfun


def test_volume():
    '''
    This function tests the volume between pol2cart
    and the fatiando function for a rectangular prism.
    
    output
    
    Assertion
    '''
    Np = 10 # number of prisms
    Nv = 8 # number of vertices

    #r = 1000. # radial distance for each vertice
    r = np.zeros(Nv)
    r[::2] = 1000.
    r[1::2] = np.sqrt(2.)*1000./2.
    
    dz = 100.0    # thickness of each prism

    area = 0.

    for i in range(Nv-1):
        area += r[i]*r[i+1]*np.sin(2.*(i+1)*np.pi/Nv - 2.*i*np.pi/Nv)
    area = area + r[-1]*r[0]*np.sin(2.*np.pi - 2.*(Nv-1)*np.pi/Nv)

    volume = area*Np*dz
    
    volume_ref = 2000.*2000.*1000.  # l*l*h
    
    assert volume == volume_ref
    
    
def test_tfa_data():
    '''
    This function tests the total field anomaly data 
    between a model from pol2cart function and the
    fatiando function for a rectangular prism.
    
    output
    
    Assertion
    '''
    Np = 10 # number of prisms
    Nv = 8 # number of vertices

    #r = 1000. # radial distance for each vertice
    r = np.zeros(Nv)
    r[::2] = 1000.
    r[1::2] = np.sqrt(2.)*1000.
    
    # Cartesian coordinates of the origin of each prism
    x0 = np.zeros(Np) 
    y0 = np.zeros(Np)
    
    dz = 100.0    # thickness of each prism
    
    inc, dec = -60., 50. # inclination and declination of regional field
    
    props={'magnetization': utils.ang2vec(3, inc, dec)} # physical property
    
    z0 = 100.0    # depth of the top the shallowest prism
    
    m = []   # list of prisms
    
    ### creating the lis of prisms
    
    for i in range(Np):
        m.append([r, x0[i], y0[i], z0 + dz*i, z0 + dz*(i + 1), props])
    
    model_polyprism = mfun.pol2cart(m, Np, Nv)
    
    #area over which the data are calculated
    #x minimum, x maximum, y minimum and y maximum
    area = [-10000, 10000, -10000, 10000] 

    #number of data along the y and x directions
    shape = (80,80)

    #total number of data
    N = shape[0]*shape[1]

    #coordinates x and y of the data
    x = np.linspace(area[0],area[1],shape[0]) # points in x
    y = np.linspace(area[2],area[3],shape[0]) # points in y
    xp,yp = np.meshgrid(x,y)    # creating mesh points
    xp = xp.ravel()
    yp = yp.ravel()

    #vertical coordinates of the data
    zp = -350. - 500.*utils.gaussian2d(xp, yp, 17000, 21000, 21000, 18500, angle=21) # relief
    
    tfat_polyprism = polyprism.tf(xp, yp, zp, model_polyprism, inc, dec)
    
    model_recprism = [mesher.Prism(-1000, 1000, -1000, 1000, 100, 1100, props)]
    
    tfat_recprism = prism.tf(xp, yp, zp, model_recprism, inc, dec)
    
    assert np.allclose(tfat_polyprism, tfat_recprism, atol=1e-05), 'The data from small rectangular prisms must be equal to a big rectangular prism'
    
