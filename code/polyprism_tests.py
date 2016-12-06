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
    
def test_tfa_fd_x0_data():
    '''
    This function tests the derivative of total field anomaly data
    between a model deslocated in x and the fd_tf_x0_polyprism
    function.
    
    output
    
    Assertion
    '''

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
    
    inc, dec = -60., 50. # inclination and declination of regional field
    
    props={'magnetization': utils.ang2vec(3, inc, dec)} # physical property
    
    z1 = 100.0    # depth of the top prism
    z2 = 1100.    # bottom of prism
    delta = 10.   # increment 

    # creating vertices
    r = np.zeros(4) + 1000.

    # origin
    x0 = 0.
    y0 = 0.

    ### creating the prisms
    m = [[r, x0, y0, z1, z2, props]]
    mp = [[r, x0 + delta, y0, z1, z2, props]]   # prism plus increment
    mm = [[r, x0 - delta, y0, z1, z2, props]]   # prism minus increment

    ### creating data of the prisms
    mpt = mfun.pol2cart(mp, len(mp), r.size)
    mmt = mfun.pol2cart(mm, len(mm), r.size)

    mp_fat = polyprism.tf(xp, yp, zp, mpt, inc, dec)   # data of prism plus increment
    mm_fat = polyprism.tf(xp, yp, zp, mmt, inc, dec)   # data of prism minus increment

    # calculating the derivatives

    df_m = mfun.fd_tf_x0_polyprism(xp, yp, zp, m[0], r.size, delta, inc, dec)  # derivative from the function
    df_mp_mm = (mp_fat - mm_fat)/(2.*delta)  # derivative from difference of data
    
    assert np.allclose(df_m, df_mp_mm), 'The derivative is not correct'
    
def test_tfa_fd_y0_data():
    '''
    This function tests the derivative of total field anomaly data
    between a model deslocated in y and the fd_tf_y0_polyprism
    function.
    
    output
    
    Assertion
    '''

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
    
    inc, dec = -60., 50. # inclination and declination of regional field
    
    props={'magnetization': utils.ang2vec(3, inc, dec)} # physical property
    
    z1 = 100.0    # depth of the top prism
    z2 = 1100.0    # bottom of prism
    delta = 10.0   # increment 

    # creating vertices
    r = np.zeros(4) + 1000.

    # origin
    x0 = 0.
    y0 = 0.

    ### creating the prisms
    m = [[r, x0, y0, z1, z2, props]]
    mp = [[r, x0, y0 + delta, z1, z2, props]]   # prism plus increment
    mm = [[r, x0, y0 - delta, z1, z2, props]]   # prism minus increment

    ### creating data of the prisms
    mpt = mfun.pol2cart(mp, len(mp), r.size)
    mmt = mfun.pol2cart(mm, len(mm), r.size)

    mp_fat = polyprism.tf(xp, yp, zp, mpt, inc, dec)   # data of prism plus increment
    mm_fat = polyprism.tf(xp, yp, zp, mmt, inc, dec)   # data of prism minus increment

    # calculating the derivatives

    df_m = mfun.fd_tf_y0_polyprism(xp, yp, zp, m[0], r.size, delta, inc, dec)  # derivative from the function
    df_mp_mm = (mp_fat - mm_fat)/(2.*delta)  # derivative from difference of data
    
    assert np.allclose(df_m, df_mp_mm), 'The derivative is not correct'
    
def test_tfa_fd_radial_data():
    '''
    This function tests the derivative of total field anomaly data
    between a model deslocated in a radial distance and the 
    fd_tf_radial_polyprism function.
    
    output
    
    Assertion
    '''

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
    
    inc, dec = -60., 50. # inclination and declination of regional field
    
    props={'magnetization': utils.ang2vec(3, inc, dec)} # physical property
    
    z1 = 100.0    # depth of the top prism
    z2 = 1100.0    # bottom of prism
    delta = 10.0   # increment
    nv = 5

    # creating vertices
    r = np.zeros(nv + 2) + 1000.
    rp = np.zeros(nv + 2) + 1000.
    rp[nv] = r[nv] + delta
    rm = np.zeros(nv + 2) + 1000.
    rm[nv] = r[nv] - delta

    # origin
    x0 = 0.
    y0 = 0.
    
    ### creating the prisms
    m = [[r, x0, y0, z1, z2, props]]
    mp = [[rp, x0, y0, z1, z2, props]]   # prism plus increment
    mm = [[rm, x0, y0, z1, z2, props]]   # prism minus increment
    
    ### creating data of the prisms
    mpt = mfun.pol2cart(mp, len(mp), rp.size)
    mmt = mfun.pol2cart(mm, len(mm), rm.size)

    mp_fat = polyprism.tf(xp, yp, zp, mpt, inc, dec)   # data of prism plus increment
    mm_fat = polyprism.tf(xp, yp, zp, mmt, inc, dec)   # data of prism minus increment

    # calculating the derivatives

    df_m = mfun.fd_tf_radial_polyprism(xp, yp, zp, m[0], r.size, nv, delta, inc, dec)  # derivative from the function
    df_mp_mm = (mp_fat - mm_fat)/(2.*delta)  # derivative from difference of data
    
    assert np.allclose(df_m, df_mp_mm, atol=1e-06), 'The derivative is not correct'
    
def test_phi_1():
    '''
    This function tests the result for the phi_1 function
    for an empty matrix.
    
    output
    
    assertion
    '''
    
    M = 5   # number of vertices
    L = 1   # number of prisms
    P = L*(M + 2) # number of parameters
    
    H = np.zeros((P, P))  # hessian for phi_1
    H_ref = np.zeros((P,P)) # hessian for comparison
    
    alpha = 1.0 # smoothness parameter
    
    H = mfun.phi_1(M, L, H, alpha) # building H
    
    # building H_ref
    for i in range(M):
        H_ref[i,i] = 2.*alpha
    for i in range(M - 1):
        H_ref[i,i+1] = -1.*alpha
        H_ref[i+1,i] = -1.*alpha
    H_ref[0,M-1] = -1*alpha
    H_ref[M-1,0] = -1*alpha
        
    assert np.allclose(H, H_ref), 'The matrices is not correct'

def test_phi_2():
    '''
    This function tests the result for the phi_2 function
    for an empty matrix.
    
    output
    
    assertion
    '''
    
    M = 3   # number of vertices
    L = 3   # number of prisms
    P = L*(M + 2) # number of parameters
    
    H = np.zeros((P, P))  # hessian for phi_1
    H_ref = np.zeros((P,P)) # hessian for comparison
    
    alpha = 1.0 # smoothness parameter
    
    H = mfun.phi_2(M, L, H, alpha) # building H
    
    for i in range(M):
        H_ref[i,i] = alpha
        H_ref[i,i+M+2] = -1.*alpha
        H_ref[i+M+2,i] = -1.*alpha
        H_ref[i+M+2,i+M+2] = 2.*alpha
        H_ref[-i-3,-i-3] = alpha
        H_ref[-i-3-M-2,-i-3] = -1.*alpha
        H_ref[-i-3,-i-3-M-2] = -1.*alpha
        
    assert np.allclose(H, H_ref), 'The matrices is not correct'
        
def test_phi_3():
    '''
    This function tests the result for the phi_3 function
    for an empty matrix.
    
    output
    
    assertion
    '''
    
    M = 3   # number of vertices
    L = 1   # number of prisms
    P = L*(M + 2) # number of parameters
    
    H = np.zeros((P, P))  # hessian for phi_1
    H_ref = np.identity(M + 2) # hessian for comparison
    
    alpha = 1.0 # smoothness parameter
    
    H = mfun.phi_3(M, L, H, alpha) # building H
        
    assert np.allclose(H, H_ref), 'The matrices is not correct'
    
def test_phi_4():
    '''
    This function tests the result for the phi_4 function
    for an empty matrix.
    
    output
    
    assertion
    '''
    
    M = 4   # number of vertices
    L = 1   # number of prisms
    P = L*(M + 2) # number of parameters
    
    H = np.zeros((P, P))  # hessian for phi_1
    H_ref = np.zeros((P,P)) # hessian for comparison
    
    alpha = 1.0 # smoothness parameter
    
    H = mfun.phi_4(M, L, H, alpha) # building H
    
    # building H_ref
    H_ref[-1,-1] = alpha
    H_ref[-2,-2] = alpha
        
    assert np.allclose(H, H_ref), 'The matrices is not correct'
    
def test_phi_5():
    '''
    This function tests the result for the phi_5 function
    for an empty matrix.
    
    output
    
    assertion
    '''
    
    M = 4   # number of vertices
    L = 1   # number of prisms
    P = L*(M + 2) # number of parameters
    
    H = np.zeros((P, P))  # hessian for phi_1
    H_ref = np.zeros((P,P)) # hessian for comparison
    
    alpha = 1.0 # smoothness parameter
    
    H = mfun.phi_5(M, L, H, alpha) # building H
    
    # building H_ref
    H_ref[-1,-1] = alpha
    H_ref[-2,-2] = alpha
        
    assert np.allclose(H, H_ref), 'The matrices is not correct'
    
def test_phi_6():
    '''
    This function tests the result for the phi_6 function
    for an empty matrix.
    
    output
    
    assertion
    '''
    
    M = 4   # number of vertices
    L = 2   # number of prisms
    P = L*(M + 2) # number of parameters
    
    H = np.zeros((P, P))  # hessian for phi_1
    H_ref = np.zeros((P,P)) # hessian for comparison
    
    alpha = 1.0 # smoothness parameter
    
    H = mfun.phi_6(M, L, H, alpha) # building H
    
    # building H_ref
    for i in range(M):
        H_ref[i,i] = alpha
        H_ref[i+M+2,i+M+2] = alpha
        
    assert np.allclose(H, H_ref), 'The matrices is not correct'