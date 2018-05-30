### Test functions for magnetic problems with polygonal prisms

import numpy as np
from fatiando import mesher, gridder, utils
from fatiando.gravmag import polyprism
from fatiando.mesher import PolygonalPrism
from fatiando.gravmag import prism
from fatiando.mesher import Prism
from fatiando.constants import CM, T2NT
import polyprism_tests as tests
import numpy.testing as npt
import mag_polyprism_functions as mfun

def test_area_points_diff_sizes():
    'x and y points with different number of elements'

    x = np.zeros(12)
    y = np.zeros(10)
    raises(AssertionError, mfun.area_polygon, x = x, y = y)

    x = np.zeros(10)
    y = np.zeros(12)
    raises(AssertionError, mfun.area_polygon, x = x, y = y)

def test_area_points_diff_shapes():
    'x and y points with different shapes'

    x = np.zeros(10)
    y = np.zeros((10,1))
    raises(AssertionError, mfun.area_polygon, x = x, y = y)

    x = np.zeros((10,1))
    y = np.zeros(10)
    raises(AssertionError, mfun.area_polygon, x = x, y = y)

def test_area_square():
    'This function tests the area calculated by sholace formula.'
    
    l = 2.    
    x = np.array([1., 1., -1., -1.])
    y = np.array([1., -1., -1., 1.])
    
    area = mfun.area_polygon(x, y)
    
    area_ref = l*l
    
    assert np.allclose(area, area_true), 'The area is not correct'
    
def test_pol2cart_points_diff_sizes():
    'x and y points with different number of elements'

    x = np.zeros(12)
    y = np.zeros(10)
    raises(AssertionError, mfun.area_polygon, x = x, y = y)

    x = np.zeros(10)
    y = np.zeros(12)
    raises(AssertionError, mfun.area_polygon, x = x, y = y)

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
    
    assert np.allclose(volume, volume_ref), 'The volume is not correct'
    
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
    
    assert np.allclose(p, p_ref), 'The result does not match with the reference'
    
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
    zp = -350. - 500.*utils.gaussian2d(xp, yp, 17000., 21000., 21000., 18500., angle=21.) # relief
    
    tfat_polyprism = polyprism.tf(xp, yp, zp, model_polyprism, inc, dec)
    
    model_recprism = [mesher.Prism(-1000., 1000., -1000., 1000., 100., 1100., props)]
    
    tfat_recprism = prism.tf(xp, yp, zp, model_recprism, inc, dec)
    
    npt.assert_almost_equal(tfat_polyprism, tfat_recprism, decimal=5), 'The data from small rectangular prisms must be equal to a big rectangular prism'
    
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
        H_ref[i,i] = 2.*alpha
    for i in range(M - 1):
        H_ref[i,i+1] = -1.*alpha
        H_ref[i+1,i] = -1.*alpha
    H_ref[0,M-1] = -1*alpha
    H_ref[M-1,0] = -1*alpha
        
    assert np.allclose(H, H_ref), 'The matrix H is not correct'

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
        H_ref[i,i] = alpha
        H_ref[i,i+M+2] = -1.*alpha
        H_ref[i+M+2,i] = -1.*alpha
        H_ref[i+M+2,i+M+2] = 2.*alpha
        H_ref[-i-3,-i-3] = alpha
        H_ref[-i-3-M-2,-i-3] = -1.*alpha
        H_ref[-i-3,-i-3-M-2] = -1.*alpha
        
    assert np.allclose(H, H_ref), 'The matrices is not correct'
        
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
        
    assert np.allclose(H, H_ref), 'The matrices is not correct'
    
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
    H_ref[-1,-1] = alpha
    H_ref[-2,-2] = alpha
        
    assert np.allclose(H, H_ref), 'The matrices is not correct'
    
def test_Hessian_phi_5():
    '''
    This function tests the result for the Hessian_phi_5 function
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
    
    H = mfun.Hessian_phi_5(M, L, H, alpha) # building H
    
    # building H_ref
    H_ref[-1,-1] = alpha
    H_ref[-2,-2] = alpha
        
    assert np.allclose(H, H_ref), 'The matrices is not correct'
    
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
        H_ref[i,i] = alpha
        H_ref[i+M+2,i+M+2] = alpha
        
    assert np.allclose(H, H_ref), 'The matrices is not correct'
    
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
    
    assert np.allclose(d0, dzero), 'The diagonal is not correct'
    assert np.allclose(d1, done), 'The diagonal is not correct'
    assert np.allclose(dM, dm), 'The diagonal is not correct'
    
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
    
    assert np.allclose(d0, dzero), 'The diagonal is not correct'
    assert np.allclose(d1, done), 'The diagonal is not correct'
    
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

    assert np.allclose(d0, dzero), 'The diagonal is not correct'
    assert np.allclose(d1, done), 'The diagonal is not correct'
    
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
    
    assert np.allclose(d0, dzero), 'The diagonal is not correct'
    assert np.allclose(d1, done), 'The diagonal is not correct'
    
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

    assert np.allclose(d0, dzero), 'The diagonal is not correct'
    assert np.allclose(d1, done), 'The diagonal is not correct'
    
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
    
    assert np.allclose(d0, dzero), 'The diagonal is not correct'
    
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
    
    assert np.allclose(grad, grad_ref), 'The gradient is not correct'
    
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
    
    assert np.allclose(grad, grad_ref), 'The gradient is not correct'
    
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
    
    
    assert np.allclose(grad, grad_ref), 'The gradient is not correct'
    
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
    
    assert np.allclose(grad, grad_ref), 'The gradient is not correct'
    
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
    
    assert np.allclose(grad, grad_ref), 'The gradient is not correct'
    
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
    
    assert np.allclose(grad, grad_ref), 'The gradient is not correct'
    
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
        
    assert np.allclose(grad, grad_ref), 'The gradient is not correct'
    
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
    
    assert np.allclose(grad, grad_ref), 'The gradient is not correct'
    
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
    
    assert np.allclose(grad, grad_ref), 'The gradient is not correct'

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
    phi_ref = 6.*alpha
    phi = mfun.phi_1(M, L, m, alpha)
    
    assert np.allclose(phi, phi_ref), 'The value of constraint is not correct'
    
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
    phi_ref = 75.*alpha
    phi = mfun.phi_2(M, L, m, alpha)
    
    assert np.allclose(phi, phi_ref), 'The value of constraint is not correct'
    
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
    
    assert np.allclose(phi, phi_ref), 'The value of constraint is not correct'
    
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
    
    assert np.allclose(phi, phi_ref), 'The value of constraint is not correct'

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
    phi_ref = 100.*alpha
    phi = mfun.phi_5(M, L, m, alpha)
        
    assert np.allclose(phi, phi_ref), 'The value of constraint is not correct'
    
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
        
    assert np.allclose(phi, phi_ref), 'The value of constraint is not correct'

def test_trans_parameter2_zeros():
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
    m = np.zeros(P)
    # limits for parameters in meters
    rmin = -4000.
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
    
    assert np.allclose(m, mt), 'The resultant vector is different of zero'
    
def test_trans_inv_parameter2_zeros():
    '''
    Test for parameter inverse transformation during
    the Levenberg-Marquadt algoithm with a known
    vector and oposite limits values for 
    the parameters
    
    output
    Assertion
    '''
    M = 8
    L = 5
    P = L*(M+2)
    m0 = np.zeros(M+2)
    m0[:M] = 3000.
    m0[M] = 100.
    m0[M+1] = 100.
    m0 = np.resize(m0, P)
    # limits for parameters in meters
    rmin = -4000.
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
    
    mt = mfun.trans_parameter2(m0, M, L, mmax, mmin)
    
    m = mfun.trans_inv_parameter2(mt, M, L, mmax, mmin)
    
    assert np.allclose(m0, m), 'The resultant vector is different from zero'