### Test functions for magnetic problems with polygonal prisms

import numpy as np
import numpy.testing as npt
from fatiando import mesher, gridder, utils
from fatiando.gravmag import polyprism
from fatiando.mesher import PolygonalPrism
from fatiando.gravmag import prism
from fatiando.mesher import Prism
from fatiando.constants import CM, T2NT
import mag_polyprism_functions as mfun

def test_volume():
    '''
    This function tests the volume between pol2cart
    and the fatiando function for a rectangular prism.
    
    output
    
    Assertion
    '''
    L = 1 # number of prisms
    M = 4 # number of vertices

    #r = 1000. # radial distance for each vertice
    r = np.zeros(M) + np.sqrt(2000000.)
        
    # Cartesian coordinates of the origin of each prism
    x0 = 0. 
    y0 = 0.
    
    dz = 1000.0    # thickness of each prism
    
    inc, dec = -60., 50. # inclination and declination of regional field
    
    props={'magnetization': utils.ang2vec(3, inc, dec)} # physical property
    
    z0 = 100.0    # depth of the top the shallowest prism
    
    m = []   # list of prisms
    
    ### creating the lis of prisms
    
    m.append([r, x0, y0, z0, z0 + dz, props])
    
    model = mfun.pol2cart(m, M, L)

    area = mfun.area_polygon(model[0].x, model[0].y)

    volume = area*L*dz
    
    volume_ref = 2000.*2000.*1000.  # l*l*h
    
    npt.assert_almost_equal(volume, volume_ref, decimal=5), 'The volume is not correct'
    
def test_paramvec():
    '''
    Test for function that transform a list of prisms into
    a parameter vector.
    
    output
    
    Assertion
    '''
    L = 2 # number of prisms
    M = 4 # number of vertices
    P = L*(M + 2) # number of parameters

    #r = 1000. # radial distance for each vertice
    r = np.zeros(M) + 1000.
        
    # Cartesian coordinates of the origin of each prism
    x0 = 0. 
    y0 = 0.
    
    dz = 100.0    # thickness of each prism
    
    inc, dec = -60., 50. # inclination and declination of regional field
    
    props={'magnetization': utils.ang2vec(3, inc, dec)} # physical property
    
    z0 = 100.0    # depth of the top the shallowest prism
    
    m = []   # list of prisms
    
    ### creating the lis of prisms
    
    for i in range(L):
        m.append([r, x0, y0, z0 + dz*i, z0 + dz*(i + 1), props])
    
    model_polyprism = mfun.pol2cart(m, M, L)
    
    p = mfun.param_vec(m, M, L)
    
    p_ref = np.zeros(P)
    p_ref[:M] = r 
    p_ref[M+2:2*M+2] = r
    
    npt.assert_almost_equal(p, p_ref), 'The result does not match with the reference'
    
def test_param2model():
    '''
    Test for function that transform a parameters vector into
    a list of prisms.
    
    output
    
    Assertion
    '''
    L = 2 # number of prisms
    M = 4 # number of vertices
    P = L*(M + 2) # number of parameters

    #r = 1000. # radial distance for each vertice
    r = np.zeros(M) + 1000.
        
    # Cartesian coordinates of the origin of each prism
    x0 = 0. 
    y0 = 0.
    
    dz = 100.0    # thickness of each prism
    
    inc, dec = -60., 50. # inclination and declination of regional field
    
    props={'magnetization': utils.ang2vec(3, inc, dec)} # physical property
    
    z0 = 100.0    # depth of the top the shallowest prism
    
    m = []   # list of prisms
    
    ### creating the lis of prisms
    
    for i in range(L):
        m.append([r, x0, y0, z0 + dz*i, z0 + dz*(i + 1), props])
        
    # transform the list of prisms into parameters vector
    p = mfun.param_vec(m, M, L)
    
    # transform the parameters vector into list of prisms
    model = mfun.param2model(p, M, L, z0, dz, props)
    
    ma = mfun.pol2cart(m,M,L)
    moda = mfun.pol2cart(model,M,L)
    
    for i in range(L):
        assert ma[i].x.all() == moda[i].x.all()
        assert ma[i].y.all() == moda[i].y.all()
        assert ma[i].z1 == moda[i].z1
        assert ma[i].z2 == moda[i].z2
        assert ma[i].props == moda[i].props

def test_tfa_data():
    '''
    This function tests the total field anomaly data 
    between a model from pol2cart function and the
    fatiando function for a rectangular prism.
    
    output
    
    Assertion
    '''
    L = 10 # number of prisms
    M = 8 # number of vertices

    #r = 1000. # radial distance for each vertice
    r = np.zeros(M)
    r[::2] = 1000.
    r[1::2] = np.sqrt(2.)*1000.
    
    # Cartesian coordinates of the origin of each prism
    x0 = np.zeros(L) 
    y0 = np.zeros(L)
    
    dz = 100.0    # thickness of each prism
    
    inc, dec = -60., 50. # inclination and declination of regional field
    
    props={'magnetization': utils.ang2vec(3, inc, dec)} # physical property
    
    z0 = 100.0    # depth of the top the shallowest prism
    
    m = []   # list of prisms
    
    ### creating the lis of prisms
    
    for i in range(L):
        m.append([r, x0[i], y0[i], z0 + dz*i, z0 + dz*(i + 1), props])
    
    model_polyprism = mfun.pol2cart(m, M, L)
    
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
    
    npt.assert_almost_equal(tfat_polyprism, tfat_recprism, decimal=5), 'The data from small rectangular prisms must be equal to a big rectangular prism'

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
    mpt = mfun.pol2cart(mp, r.size, len(mp))
    mmt = mfun.pol2cart(mm, r.size, len(mm))

    mp_fat = polyprism.tf(xp, yp, zp, mpt, inc, dec)   # data of prism plus increment
    mm_fat = polyprism.tf(xp, yp, zp, mmt, inc, dec)   # data of prism minus increment

    # calculating the derivatives

    df_m = mfun.fd_tf_x0_polyprism(xp, yp, zp, m[0], r.size, delta, inc, dec)  # derivative from the function
    df_mp_mm = (mp_fat - mm_fat)/(2.*delta)  # derivative from difference of data
    
    npt.assert_almost_equal(df_m, df_mp_mm), 'The derivative is not correct'
    
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
    mpt = mfun.pol2cart(mp, r.size, len(mp))
    mmt = mfun.pol2cart(mm, r.size, len(mm))

    mp_fat = polyprism.tf(xp, yp, zp, mpt, inc, dec)   # data of prism plus increment
    mm_fat = polyprism.tf(xp, yp, zp, mmt, inc, dec)   # data of prism minus increment

    # calculating the derivatives

    df_m = mfun.fd_tf_y0_polyprism(xp, yp, zp, m[0], r.size, delta, inc, dec)  # derivative from the function
    df_mp_mm = (mp_fat - mm_fat)/(2.*delta)  # derivative from difference of data
        
    npt.assert_almost_equal(df_m, df_mp_mm), 'The derivative is not correct'
    
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
    rp = r.copy()
    rp[nv] += delta
    rm = r.copy()
    rm[nv] -= delta

    # origin
    x0 = 0.
    y0 = 0.
    
    ### creating the prisms
    m = [[r, x0, y0, z1, z2, props]]
    mp = [[rp, x0, y0, z1, z2, props]]   # prism plus increment
    mm = [[rm, x0, y0, z1, z2, props]]   # prism minus increment
    
    ### creating data of the prisms
    mpt = mfun.pol2cart(mp, rp.size, len(mp))
    mmt = mfun.pol2cart(mm, rm.size, len(mm))

    mp_fat = polyprism.tf(xp, yp, zp, mpt, inc, dec)   # data of prism plus increment
    mm_fat = polyprism.tf(xp, yp, zp, mmt, inc, dec)   # data of prism minus increment

    # calculating the derivatives

    df_m = mfun.fd_tf_radial_polyprism(xp, yp, zp, m[0], r.size, nv, delta, inc, dec)  # derivative from the function
    df_mp_mm = (mp_fat - mm_fat)/(2.*delta)  # derivative from difference of data
    
    npt.assert_almost_equal(df_m, df_mp_mm, decimal=5), 'The derivative is not correct'
    
def test_Hessian_phi_1():
    '''
    This function tests the result for the Hessian_phi_1 function
    for an empty matrix.
    
    output
    
    assertion
    '''
    
    M = 5   # number of vertices
    L = 1   # number of prisms
    P = L*(M + 2) # number of parameters
    
    H = np.zeros((P, P))  # hessian for phi_1
    H_ref = np.zeros((P,P)) # hessian for comparison
    
    alpha = 1.0 # regularization parameter
    
    H = mfun.Hessian_phi_1(M, L, H, alpha) # building H
    
    # building H_ref
    for i in range(M):
        H_ref[i,i] += 2.*alpha
    for i in range(M - 1):
        H_ref[i,i+1] -= alpha
        H_ref[i+1,i] -= alpha
    H_ref[0,M-1] -= alpha
    H_ref[M-1,0] -= alpha
        
    npt.assert_almost_equal(H, H_ref), 'The matrix H is not correct'

def test_Hessian_phi_2():
    '''
    This function tests the result for the Hessian_phi_2 function
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
    
    H = mfun.Hessian_phi_2(M, L, H, alpha) # building H
    
    for i in range(M):
        H_ref[i,i] += alpha
        H_ref[i,i+M+2] -= alpha
        H_ref[i+M+2,i] -= alpha
        H_ref[i+M+2,i+M+2] += 2.*alpha
        H_ref[-i-3,-i-3] += alpha
        H_ref[-i-3-M-2,-i-3] -= alpha
        H_ref[-i-3,-i-3-M-2] -= alpha
        
    npt.assert_almost_equal(H, H_ref), 'The matrices is not correct'
        
def test_Hessian_phi_3():
    '''
    This function tests the result for the Hessian_phi_3 function
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
    
    H = mfun.Hessian_phi_3(M, L, H, alpha) # building H
        
    npt.assert_almost_equal(H, H_ref), 'The matrices is not correct'
    
def test_Hessian_phi_4():
    '''
    This function tests the result for the Hessian_phi_4 function
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
    
    H = mfun.Hessian_phi_4(M, L, H, alpha) # building H
    
    # building H_ref
    H_ref[-1,-1] += alpha
    H_ref[-2,-2] += alpha
        
    npt.assert_almost_equal(H, H_ref), 'The matrices is not correct'
    
def test_Hessian_phi_5():
    '''
    This function tests the result for the Hessian_phi_5 function
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
    
    H = mfun.Hessian_phi_5(M, L, H, alpha) # building H
    
    # building H_ref
    H_ref[M,M] += alpha
    H_ref[M+1,M+1] += alpha
    H_ref[2*M+2,M] -= alpha
    H_ref[2*M+3,M+1] -= alpha
    H_ref[M,2*M+2] -= alpha
    H_ref[M+1,2*M+3] -= alpha
    H_ref[-1,-1] += alpha
    H_ref[-2,-2] += alpha
        
    npt.assert_almost_equal(H, H_ref), 'The matrices is not correct'
    
def test_Hessian_phi_6():
    '''
    This function tests the result for the Hessian_phi_6 function
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
    
    H = mfun.Hessian_phi_6(M, L, H, alpha) # building H
    
    # building H_ref
    for i in range(M):
        H_ref[i,i] += alpha
        H_ref[i+M+2,i+M+2] += alpha
        
    npt.assert_almost_equal(H, H_ref), 'The matrices is not correct'
    
def test_diags_phi_1():
    '''
    This function tests the result for the diags_phi_1 function
    for an simple example.
    
    output
    
    assertion
    '''
    M = 4   # number of vertices
    L = 2   # number of prisms
    P = L*(M + 2) # number of parameters
    alpha = 1. # regularization
    
    d0, d1, dM = mfun.diags_phi_1(M, L, alpha) # non-zero diagonals
    
    dzero = np.array([2.*alpha, 2.*alpha, 2.*alpha, 2.*alpha, 0., 0.])
    dzero = np.resize(dzero, P)
    done = np.array([-alpha, -alpha, -alpha, 0., 0., 0.])
    done = np.resize(done, P-1)
    dm = np.array([-alpha, 0., 0., 0., 0., 0.])
    dm = np.resize(dm, P-M+1)
    
    npt.assert_almost_equal(d0, dzero), 'The diagonal is not correct'
    npt.assert_almost_equal(d1, done), 'The diagonal is not correct'
    npt.assert_almost_equal(dM, dm), 'The diagonal is not correct'
    
def test_diags_phi_2_tp():
    '''
    This function tests the result for the diags_phi_2 function
    for an simple example with two prisms.
    
    output
    
    assertion
    '''
    M = 4   # number of vertices
    L = 2   # number of prisms
    P = L*(M + 2) # number of parameters
    alpha = 1. # regularization
    
    d0, d1 = mfun.diags_phi_2(M, L, alpha) # non-zero diagonals
    
    dzero = np.array([alpha, alpha, alpha, alpha, 0., 0.])
    dzero = np.resize(dzero, P)
    done = np.array([-alpha, -alpha, -alpha, -alpha, 0., 0.])
    done = np.resize(done, P-M-2)
    
    npt.assert_almost_equal(d0, dzero), 'The diagonal is not correct'
    npt.assert_almost_equal(d1, done), 'The diagonal is not correct'
    
def test_diags_phi_2_mp():
    '''
    This function tests the result for the diags_phi_2 function
    for an simple example with more than two prisms.
    
    output
    
    assertion
    '''
    M = 4   # number of vertices
    L = 3   # number of prisms
    P = L*(M + 2) # number of parameters
    alpha = 1. # regularization
    
    d0, d1 = mfun.diags_phi_2(M, L, alpha) # non-zero diagonals
    
    dzero = np.array([alpha, alpha, alpha, alpha, 0., 0.])
    dzero = np.resize(dzero, P)
    dzero[M+2:2*M+2] += alpha
    done = np.array([-alpha, -alpha, -alpha, -alpha, 0., 0.])
    done = np.resize(done, P-M-2)

    npt.assert_almost_equal(d0, dzero), 'The diagonal is not correct'
    npt.assert_almost_equal(d1, done), 'The diagonal is not correct'
    
def test_diags_phi_5_tp():
    '''
    This function tests the result for the diags_phi_5 function
    for an simple example with two prisms.
    
    output
    
    assertion
    '''
    M = 4   # number of vertices
    L = 2   # number of prisms
    P = L*(M + 2) # number of parameters
    alpha = 1. # regularization
    
    d0, d1 = mfun.diags_phi_5(M, L, alpha) # non-zero diagonals
    
    dzero = np.array([0., 0., 0., 0., alpha, alpha])
    dzero = np.resize(dzero, P)
    done = np.array([0., 0., 0., 0., -alpha, -alpha])
    done = np.resize(done, P-M-2)
    
    npt.assert_almost_equal(d0, dzero), 'The diagonal is not correct'
    npt.assert_almost_equal(d1, done), 'The diagonal is not correct'
    
def test_diags_phi_5_mp():
    '''
    This function tests the result for the diags_phi_5 function
    for an simple example with more than two prisms.
    
    output
    
    assertion
    '''
    M = 4   # number of vertices
    L = 3   # number of prisms
    P = L*(M + 2) # number of parameters
    alpha = 1. # regularization
    
    d0, d1 = mfun.diags_phi_5(M, L, alpha) # non-zero diagonals
    
    dzero = np.array([0., 0., 0., 0., alpha, alpha])
    dzero = np.resize(dzero, P)
    dzero[2*M+2:2*(M+2)] += alpha
    done = np.array([0., 0., 0., 0., -alpha, -alpha])
    done = np.resize(done, P-M-2)

    npt.assert_almost_equal(d0, dzero), 'The diagonal is not correct'
    npt.assert_almost_equal(d1, done), 'The diagonal is not correct'
    
def test_diags_phi_6():
    '''
    This function tests the result for the diags_phi_6 function
    for an simple example.
    
    output
    
    assertion
    '''
    M = 4   # number of vertices
    L = 2   # number of prisms
    P = L*(M + 2) # number of parameters
    alpha = 1. # regularization
    
    d0 = mfun.diags_phi_6(M, L, alpha) # non-zero diagonals
    
    dzero = np.array([alpha, alpha, alpha, alpha, 0., 0.])
    dzero = np.resize(dzero, P)
    
    npt.assert_almost_equal(d0, dzero), 'The diagonal is not correct'
    
def test_gradient_phi_1_unitary():
    '''
    This function tests the result for the gradient_phi_1 function
    for an unitary vector.
    
    output
    
    assertion
    '''
    M = 3   # number of vertices
    L = 1   # number of prisms
    P = L*(M + 2) # number of parameters
    alpha = .1 # regularization
    m = np.ones(P)*5 # gradient
    grad_ref = m.copy()    
    grad = mfun.gradient_phi_1(M, L, m, alpha)
    
    npt.assert_almost_equal(grad, grad_ref), 'The gradient is not correct'
    
def test_gradient_phi_1_arranged():
    '''
    This function tests the result for the gradient_phi_1 function
    for an arranged vector.
    
    output
    
    assertion
    '''
    M = 3   # number of vertices
    L = 1   # number of prisms
    P = L*(M + 2) # number of parameters
    alpha = .01 # regularization
    m = np.arange(1., 6., 1.) # gradient
    grad_ref = m.copy()
    grad_ref[0] -= 3.*alpha
    grad_ref[2] += 3.*alpha
    grad = mfun.gradient_phi_1(M, L, m, alpha)
    
    npt.assert_almost_equal(grad, grad_ref), 'The gradient is not correct'
    
def test_gradient_phi_2_unitary():
    '''
    This function tests the result for the gradient_phi_2 function
    for an unitary vector.
    
    output
    
    assertion
    '''
    M = 3   # number of vertices
    L = 2   # number of prisms
    P = L*(M + 2) # number of parameters
    alpha = .1 # regularization
    m = np.ones(P)*5 # gradient
    grad_ref = m.copy()    
    grad = mfun.gradient_phi_2(M, L, m, alpha)
    
    
    npt.assert_almost_equal(grad, grad_ref), 'The gradient is not correct'
    
def test_gradient_phi_2_arranged():
    '''
    This function tests the result for the gradient_phi_2 function
    for an arranged vector.
    
    output
    
    assertion
    '''
    M = 3   # number of vertices
    L = 2   # number of prisms
    P = L*(M + 2) # number of parameters
    alpha = 1. # regularization
    m = np.arange(1., P+1., 1.) # gradient
    grad_ref = m.copy()
    grad_ref[:M] -= 5.*alpha
    grad_ref[M+2:-2] += 5.*alpha
    grad = mfun.gradient_phi_2(M, L, m, alpha)
    
    npt.assert_almost_equal(grad, grad_ref), 'The gradient is not correct'
    
def test_gradient_phi_3():
    '''
    This function tests the result for the gradient_phi_3 function.
    
    output
    
    assertion
    '''
    M = 3   # number of vertices
    L = 2   # number of prisms
    P = L*(M + 2) # number of parameters
    alpha = 1. # regularization
    m = np.arange(1., P+1., 1.) # gradient
    m0 = np.arange(1., M+3., 1.) # parameters of outcropping body
    grad_ref = m.copy()
    grad_ref[:M+2] += (grad_ref[:M+2] - m0)*alpha
    grad = mfun.gradient_phi_3(M, L, m, m0, alpha)
    
    npt.assert_almost_equal(grad, grad_ref), 'The gradient is not correct'
    
def test_gradient_phi_4():
    '''
    This function tests the result for the gradient_phi_4 function.
    
    output
    
    assertion
    '''
    M = 3   # number of vertices
    L = 2   # number of prisms
    P = L*(M + 2) # number of parameters
    alpha = 1. # regularization
    m = np.arange(1., P+1., 1.) # gradient
    m0 = np.arange(1., 3., 1.) # parameters of outcropping body
    grad_ref = m.copy()
    grad_ref[M:M+2] += (grad_ref[M:M+2] - m0)*alpha
    grad = mfun.gradient_phi_4(M, L, m, m0, alpha)
    
    npt.assert_almost_equal(grad, grad_ref), 'The gradient is not correct'
    
def test_gradient_phi_5_unitary():
    '''
    This function tests the result for the gradient_phi_5 function
    for an unitary vector.
    
    output
    
    assertion
    '''
    M = 3   # number of vertices
    L = 3   # number of prisms
    P = L*(M + 2) # number of parameters
    alpha = .1 # regularization
    m = np.ones(P)*5 # gradient
    grad_ref = m.copy()
    grad = mfun.gradient_phi_5(M, L, m, alpha)
        
    npt.assert_almost_equal(grad, grad_ref), 'The gradient is not correct'
    
def test_gradient_phi_5_arranged():
    '''
    This function tests the result for the gradient_phi_5 function
    for an arranged vector.
    
    output
    
    assertion
    '''
    M = 3   # number of vertices
    L = 3   # number of prisms
    P = L*(M + 2) # number of parameters
    alpha = 1. # regularization
    m = np.arange(1., P+1., 1.) # gradient
    grad_ref = m.copy()
    grad_ref[M:M+2] -= 5.*alpha
    grad_ref[2*(M+2)+M:] += 5.*alpha
    grad = mfun.gradient_phi_5(M, L, m, alpha)
    
    npt.assert_almost_equal(grad, grad_ref), 'The gradient is not correct'
    
def test_gradient_phi_6():
    '''
    This function tests the result for the gradient_phi_6 function
    for an unitary vector.
    
    output
    
    assertion
    '''
    M = 3   # number of vertices
    L = 2   # number of prisms
    P = L*(M + 2) # number of parameters
    alpha = 1. # regularization
    m = np.ones(P)*5. # gradient
    grad_ref = m.copy() + m*alpha
    grad_ref[M:M+2] -= 5.
    grad_ref[2*M+2:] -= 5.
    grad = mfun.gradient_phi_6(M, L, m, alpha)
    
    npt.assert_almost_equal(grad, grad_ref), 'The gradient is not correct'

def test_phi_1_arranged():
    '''
    This function tests the result for the phi_1 function
    for an arranged vector.
    
    output
    
    assertion
    '''
    M = 3   # number of vertices
    L = 1   # number of prisms
    P = L*(M + 2) # number of parameters
    alpha = .01 # regularization
    m = np.arange(1., P+1., 1.) # gradient
    phi_ref = (M-1)*M*L*alpha
    phi = mfun.phi_1(M, L, m, alpha)
    
    npt.assert_almost_equal(phi, phi_ref), 'The value of constraint is not correct'
    
def test_phi_2_arranged():
    '''
    This function tests the result for the phi_2 function
    for an arranged vector.
    
    output
    
    assertion
    '''
    M = 3   # number of vertices
    L = 2   # number of prisms
    P = L*(M + 2) # number of parameters
    alpha = .01 # regularization
    m = np.arange(1., P+1., 1.) # gradient
    phi_ref = M*(M+2)*(M+2)*alpha
    phi = mfun.phi_2(M, L, m, alpha)
    
    npt.assert_almost_equal(phi, phi_ref), 'The value of constraint is not correct'
    
def test_phi_3_arranged():
    '''
    This function tests the result for the phi_3 function.
    
    output
    
    assertion
    '''
    M = 3   # number of vertices
    L = 2   # number of prisms
    P = L*(M + 2) # number of parameters
    alpha = 1. # regularization
    m = np.arange(5., P+5., 1.) # gradient
    m0 = np.arange(1., M+3., 1.) # parameters of outcropping body
    m3 = (m[:M+2] - m0)*alpha
    phi_ref = np.sum(m3*m[:M+2])
    phi = mfun.phi_3(M, L, m, m0, alpha)
    
    npt.assert_almost_equal(phi, phi_ref), 'The value of constraint is not correct'
    
def test_phi_4_arranged():
    '''
    This function tests the result for the phi_4 function.
    
    output
    
    assertion
    '''
    M = 3   # number of vertices
    L = 2   # number of prisms
    P = L*(M + 2) # number of parameters
    alpha = 1. # regularization
    m = np.arange(5., P+5., 1.) # gradient
    m0 = np.arange(1., 3., 1.) # parameters of outcropping body
    m4 = (m[M:M+2] - m0)*alpha
    phi_ref = np.sum(m4*m[M:M+2])
    phi = mfun.phi_4(M, L, m, m0, alpha)
    
    npt.assert_almost_equal(phi, phi_ref), 'The value of constraint is not correct'

def test_phi_5_arranged():
    '''
    This function tests the result for the phi_5 function
    for an arranged vector.
    
    output
    
    assertion
    '''
    M = 3   # number of vertices
    L = 3   # number of prisms
    P = L*(M + 2) # number of parameters
    alpha = .1 # regularization
    m = np.arange(5., P+5., 1.) # gradient
    phi_ref = (M+2)*(M+2)*(L+1)*alpha
    phi = mfun.phi_5(M, L, m, alpha)
        
    npt.assert_almost_equal(phi, phi_ref), 'The value of constraint is not correct'
    
def test_phi_6_arranged():
    '''
    This function tests the result for the phi_6 function
    for an arranged vector.
    
    output
    
    assertion
    '''
    M = 3   # number of vertices
    L = 3   # number of prisms
    P = L*(M + 2) # number of parameters
    alpha = 1. # regularization
    m = np.arange(1., P+1., 1.) # gradient
    phi_ref = 597.*alpha
    phi = mfun.phi_6(M, L, m, alpha)
        
    npt.assert_almost_equal(phi, phi_ref), 'The value of constraint is not correct'
    
def test_Hessian_symetry():
    '''
    This function tests the symetry of the Hessian matrix.
    
    output
    
    Assertion
    '''
    L = 1 # number of prisms
    M = 4 # number of vertices
    P = L*(M + 2) # number of parameters

    #r = 1000. # radial distance for each vertice
    r = np.zeros(M)
    r[::2] = 1000.
    r[1::2] = np.sqrt(2.)*1000./2.
    
    # Cartesian coordinates of the origin of each prism
    x0 = np.zeros(L) + 1000.
    y0 = np.zeros(L) - 1000.
    
    dz = 100.0    # thickness of each prism
    
    inc, dec = -60., 50. # inclination and declination of regional field
    
    props={'magnetization': utils.ang2vec(3, inc, dec)} # physical property
    
    z0 = 100.0    # depth of the top the shallowest prism
    
    m = []   # list of prisms
    
    ### creating the lis of prisms
    
    for i in range(L):
        m.append([r, x0[i], y0[i], z0 + dz*i, z0 + dz*(i + 1), props])
    
    model_polyprism = mfun.pol2cart(m, M, L)
    
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
    
    # increment for derivatives
    delta = 10.

    #predict data
    d_fat = polyprism.tf(xp, yp, zp, model_polyprism, inc, dec)
    
    # sensibility matrx
    A = mfun.fd_tf_sm_polyprism(xp, yp, zp, m, M, L, delta, delta, delta, inc, dec)
    
    #Hessian matrix
    H = np.dot(A.T, A)
    
    for i in range(P):
        npt.assert_almost_equal(H[i,i+1:], H[i+1:,i]), 'The sensibility matrix is not correct'

def test_trans_parameter2():
    '''
    Test for parameter transformation during the Levenberg-Marquadt
    algoithm with a vector of zeros and oposite limits values for 
    the parameters
    
    output
    Assertion
    '''
    M = 8
    L = 5
    P = L*(M+2)
    m = np.zeros(P) + 2000.
    # limits for parameters in meters
    rmin = 0.
    rmax = 4000.
    x0min = -4000.
    x0max = 4000.
    y0min = -4000.
    y0max = 4000.
    
    mmax = np.zeros(M+2)
    mmin = np.zeros(M+2)

    mmax[:M] = rmax
    mmax[M] = x0max
    mmax[M+1] = y0max
    mmin[:M] = rmin
    mmin[M] = x0min
    mmin[M+1] = y0min

    mmax = np.resize(mmax, P)
    mmin = np.resize(mmin, P)
    
    mt = mfun.trans_parameter2(m, M, L, mmax, mmin)
    
    mref = np.zeros(M+2)
    mref[:M] = 0.
    mref[M:M+2] = -np.log(2000./6000.)
    mref = np.resize(mref, P)
    
    npt.assert_almost_equal(mref, mt), 'The resultant vector is different from reference'