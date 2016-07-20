### Functions for magnetic problems with spherical simetry


import numpy as np
from fatiando import mesher, gridder, utils
from fatiando.gravmag import sphere
from fatiando.mesher import Sphere
from fatiando.vis import mpl
from fatiando.constants import CM, T2NT

### Kernel functions (second derivatives)

def kernelxx_sphere(x,y,z,m):
    '''
    This function calculates the second derivatives of a radial function.
    
    input
    
    x: float - number of points in x
    y: float - number of poitns in y
    z: float - number of poitns in z
    m: [x0,y0,z0,R,magnetization]    
    
    output
    
    kxx: float - second derivatives in x
    '''
  
    assert x.size == y.size == z.size, 'The number of points in x, y and z must be equal'

    rs = np.sqrt((x-m[0])**2 +(y-m[1])**2 + (z-m[2])**2)
    kxx = (4./3.)*np.pi*(m[3]*m[3]*m[3])*(3.*(m[0]-x)*(m[0]-x)-rs*rs)/(rs*rs*rs*rs*rs)
            
    return kxx

def kernelxy_sphere(x,y,z,m):
    '''
    This function calculates the second derivatives of a radial function.
    
    input
    
    x: float - number of points in x
    y: float - number of poitns in y
    z: float - number of poitns in z
    m: [x0,y0,z0,R,magnetization]    
    
    output
    
    kxy: float - second derivatives in x and y
    '''
 
    assert x.size == y.size == z.size, 'The number of points in x, y and z must be equal'
        
    rs = np.sqrt((x-m[0])**2 +(y-m[1])**2 + (z-m[2])**2)
    kxy = (4./3.)*np.pi*(m[3]*m[3]*m[3])*(3.*(m[0]-x)*(m[1]-y))/(rs*rs*rs*rs*rs)


    return kxy

def kernelxz_sphere(x,y,z,m):
    '''
    This function calculates the second derivatives of a radial function.
    
    input
    
    x: float - number of points in x
    y: float - number of poitns in y
    z: float - number of poitns in z
    m: [x0,y0,z0,R,magnetization]    
    
    output
    
    kxz: float - second derivatives in x and z
    '''

    assert x.size == y.size == z.size, 'The number of points in x, y and z must be equal'
        
    rs = np.sqrt((x-m[0])**2 +(y-m[1])**2 + (z-m[2])**2)
    kxz = (4./3.)*np.pi*(m[3]*m[3]*m[3])*(3.*(m[0]-x)*(m[2]-z))/(rs*rs*rs*rs*rs)

            
    return kxz

def kernelyy_sphere(x,y,z,m):
    '''
    This function calculates the second derivatives of a radial function.
    
    input
    
    x: float - number of points in x
    y: float - number of poitns in y
    z: float - number of poitns in z
    model: [x0,y0,z0,R,magnetization]    
    
    output
    
    kyy: float - second derivatives in y
    '''
    
    assert x.size == y.size == z.size, 'The number of points in x, y and z must be equal'
        
    rs = np.sqrt((x-m[0])**2 +(y-m[1])**2 + (z-m[2])**2)
    kyy = (4./3.)*np.pi*(m[3]*m[3]*m[3])*(3.*(m[1]-y)*(m[1]-y)-rs*rs)/(rs*rs*rs*rs*rs)
            
    return  kyy

def kernelyz_sphere(x,y,z,m):
    '''
    This function calculates the second derivatives of a radial function.
    
    input
    
    x: float - number of points in x
    y: float - number of poitns in y
    z: float - number of poitns in z
    model: [x0,y0,z0,R,magnetization]    
    
    output
    
    kyz: float - second derivatives in y and z
    '''

    assert x.size == y.size == z.size, 'The number of points in x, y and z must be equal'
        
    rs = np.sqrt((x-m[0])**2 +(y-m[1])**2 + (z-m[2])**2)
    kyz = (4./3.)*np.pi*(m[3]*m[3]*m[3])*(3.*(m[1]-y)*(m[2]-z))/(rs*rs*rs*rs*rs)
            
    return kyz

def kernelzz_sphere(x,y,z,m):
    '''
    This function calculates the second derivatives of a radial function.
    
    input
    
    x: float - number of points in x
    y: float - number of poitns in y
    z: float - number of poitns in z
    model: [x0,y0,z0,R,magnetization]    
    
    output
    
    kzz: float - second derivatives in z
    '''

    assert x.size == y.size == z.size, 'The number of points in x, y and z must be equal'
        
    rs = np.sqrt((x-m[0])**2 +(y-m[1])**2 + (z-m[2])**2)
    kzz = (4./3.)*np.pi*(m[3]*m[3]*m[3])*(3.*(m[2]-z)*(m[2]-z)-rs*rs)/(rs*rs*rs*rs*rs)
            
    return kzz


### Magnetic Induction functions

def bx_sphere(x,y,z,model):
    '''
    This function calculates the induction magnetic field of a shperes.
    
    input
    
    x: float - number of points in x
    y: float - number of poitns in y
    z: float - number of poitns in z
    model: list of elements [x0,y0,z0,R,magnetization]    
    
    output
    
    Bx: float - x component of the induction magnetic field
    '''
    
    Bx = np.zeros_like(x)

    assert x.size == y.size == z.size, 'The number of points in x, y and z must be equal'
        
    for m in model:
        mx, my, mz = m[4]['magnetization']
        
        Bx += mx*kernelxx_sphere(x,y,z,m) + \
              my*kernelxy_sphere(x,y,z,m) + \
              mz*kernelxz_sphere(x,y,z,m)
        
    Bx *= CM*T2NT
    return Bx

def by_sphere(x,y,z,model):
    '''
    This function calculates the induction magnetic field of a shperes.
    
    input
    
    x: float - number of points in x
    y: float - number of poitns in y
    z: float - number of poitns in z
    model: [x0,y0,z0,R,magnetization]    
    
    output
    
    By: float - y component of the induction magnetic field
    '''
    
    By = np.zeros_like(x)

    assert x.size == y.size == z.size, 'The number of points in x, y and z must be equal'
        
    for m in model:
        mx, my, mz = m[4]['magnetization']
        
        By += mx*kernelxy_sphere(x,y,z,m) + \
              my*kernelyy_sphere(x,y,z,m) + \
              mz*kernelyz_sphere(x,y,z,m)
        
    By *= CM*T2NT
    return By

def bz_sphere(x,y,z,model):
    '''
    This function calculates the induction magnetic field of a shperes.
    
    input
    
    x: float - number of points in x
    y: float - number of poitns in y
    z: float - number of poitns in z
    model: [x0,y0,z0,R,magnetization]    

    output
    
    Bz: float - z component of the induction magnetic field
    '''
    
    Bz = np.zeros_like(x)

    assert x.size == y.size == z.size, 'The number of points in x, y and z must be equal'
        
    for m in model:
        mx, my, mz = m[4]['magnetization']
        
        Bz += mx*kernelxz_sphere(x,y,z,m) + \
              my*kernelyz_sphere(x,y,z,m) + \
              mz*kernelzz_sphere(x,y,z,m)
        
    Bz *= CM*T2NT
    return Bz

### Total Field Anomaly functions

def tf_sphere(x,y,z,model,inc,dec):
    '''
    This function calculates the total field anomaly of a magnetized shperes.
    
    input
    
    x: float - number of points in x
    y: float - number of poitns in y
    z: float - number of poitns in z
    model: [x0,y0,z0,R,magnetization]    

    output
    
    tf: float/array - total field anomaly for each observation points
    '''
    
    fx, fy, fz = utils.ang2vec(1,inc,dec) # regional direction
    
    assert x.size == y.size == z.size, 'The number of points in x, y and z must be equal'
        
    tf = fx*bx_sphere(x,y,z,model) + fy*by_sphere(x,y,z,model) + fz*bz_sphere(x,y,z,model)
        
    return tf


def sm_sphere(x,y,z,model,inc,dec):
    '''
    This function calculates the sensibility matrix and the parameters vectors
	of a total field anomaly foward problem with magnetized shperes.
    
    input
    
    x: float/array - number of points in x
    y: float/array - number of poitns in y
    z: float/array - number of poitns in z
    model: [x0,y0,z0,R,magnetization] - list of spheres
	inc: float - inclination angle
	dec: float - declination angle

    output
    
    A: float/array - sensibility matrix of foward problem
	vp: float/array - parameters vector
    '''
    
    A = np.zeros((len(x),len(model)))
    vp = np.ones(len(model))

    fx, fy, fz = utils.ang2vec(1.,inc,dec) # regional direction
    
    assert x.size == y.size == z.size, 'The number of points in x, y and z must be equal'

    for j, m in enumerate (model):
        mx, my, mz = m[4]['magnetization']/np.linalg.norm(m[4]['magnetization'])
    
        A[:,j] = kernelxx_sphere(x,y,z,m)*(fx*mx - fz*mz) + \
                 kernelxy_sphere(x,y,z,m)*(fy*mx + fx*my) + \
             	 kernelxz_sphere(x,y,z,m)*(fz*mx + fx*mz) + \
             	 kernelyy_sphere(x,y,z,m)*(fy*my - fz*mz) + \
                 kernelyz_sphere(x,y,z,m)*(fz*my + fy*mz)
        
        vp[j] = vp[j]*np.linalg.norm(m[4]['magnetization']) 

    A *= CM*T2NT
	
    return A, vp