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
    inc: float - inclination of the local-geomagnetic field
    dec: float - declination of the local-geomagnetic field
    
    output
    
    tf: float/array - total field anomaly for each observation points
    '''
    
    fx, fy, fz = utils.ang2vec(1,inc,dec) # regional direction
    
    assert x.size == y.size == z.size, 'The number of points in x, y and z must be equal'
        
    tf = fx*bx_sphere(x,y,z,model) + fy*by_sphere(x,y,z,model) + fz*bz_sphere(x,y,z,model)
        
    return tf


def sm_tf_sphere(x, y, z, xs, ys, zs, inc, dec, incs, decs):
    '''
    This function calculates the sensibility matrix and the parameter vectors
    of a total field anomaly foward problem with magnetized shperes.
    
    input
    
    x: float/array - number of points in x
    y: float/array - number of poitns in y
    z: float/array - number of poitns in z
    xs: float/array - source position
    ys: float/array - source position
    zs: float/array - source position
    inc: float - inclination of the local-geomagnetic field
    dec: float - declination of the local-geomagnetic field
    incs: float - inclination of the sources
    decs: float - declination of the sourcesces

    output
    
    A: float/array - sensibility matrix of foward problem
    '''
    
    assert x.size == y.size == z.size, 'The number of points in x, y and z must be equal'
    assert xs.size == ys.size == zs.size, 'The number of points in xs, ys and zs must be equal'
    
    A = np.empty((x.size,xs.size))
    
    fx, fy, fz = utils.ang2vec(1.,inc,dec) # regional direction
    mx, my, mz = utils.ang2vec(1.,incs,decs) # sources direction
    R = (3./(4.*np.pi))**(1./3)

    for j, (xf, yf, zf) in enumerate(zip(xs,ys,zs)):
        m = [xf, yf, zf, R, 1.]
    
        A[:,j] = kernelxx_sphere(x,y,z,m)*(fx*mx - fz*mz) + \
                 kernelxy_sphere(x,y,z,m)*(fy*mx + fx*my) + \
                 kernelxz_sphere(x,y,z,m)*(fz*mx + fx*mz) + \
                 kernelyy_sphere(x,y,z,m)*(fy*my - fz*mz) + \
                 kernelyz_sphere(x,y,z,m)*(fz*my + fy*mz)
        
    A *= CM*T2NT
    return A


def sm_bx_sphere(x, y, z, xs, ys, zs, incs, decs):
    '''
    This function calculates the sensibility matrix
    of the x component from the magnetic induction field
    for the foward problem with magnetic spheres.
    
    input
    
    x: float/array - number of points in x
    y: float/array - number of poitns in y
    z: float/array - number of poitns in z
    xs: float/array - source position
    ys: float/array - source position
    zs: float/array - source position
    incs: float - inclination of the sources
    decs: float - declination of the sources

    output
    
    A: float/array - sensibility matrix of foward problem
    '''
    
    assert x.size == y.size == z.size, 'The number of points in x, y and z must be equal'
    assert xs.size == ys.size == zs.size, 'The number of points in xs, ys and zs must be equal'
    
    A = np.empty((x.size,xs.size))
    
    mx, my, mz = utils.ang2vec(1.,incs,decs) # sources direction
    R = (3./(4.*np.pi))**(1./3)

    for j, (xf, yf, zf) in enumerate(zip(xs,ys,zs)):
        m = [xf, yf, zf, R]
    
        A[:,j] = kernelxx_sphere(x,y,z,m)*mx + \
                 kernelxy_sphere(x,y,z,m)*my + \
                 kernelxz_sphere(x,y,z,m)*mz
    A *= CM*T2NT
    return A

def sm_by_sphere(x, y, z, xs, ys, zs, incs, decs):
    '''
    This function calculates the sensibility matrix
    of the y component from the magnetic induction field
    for the foward problem with magnetic spheres.
    
    input
    
    x: float/array - number of points in x
    y: float/array - number of poitns in y
    z: float/array - number of poitns in z
    xs: float/array - source position
    ys: float/array - source position
    zs: float/array - source position
    incs: float - inclination of the sources
    decs: float - declination of the sourceses

    output
    
    A: float/array - sensibility matrix of foward problem
    '''
    
    assert x.size == y.size == z.size, 'The number of points in x, y and z must be equal'
    assert xs.size == ys.size == zs.size, 'The number of points in xs, ys and zs must be equal'
    
    A = np.empty((x.size,xs.size))
    
    mx, my, mz = utils.ang2vec(1.,incs,decs) # sources direction
    R = (3./(4.*np.pi))**(1./3)

    for j, (xf, yf, zf) in enumerate(zip(xs,ys,zs)):
        m = [xf, yf, zf, R]
                
                
        A[:,j] = kernelxy_sphere(x,y,z,m)*mx + \
                 kernelyy_sphere(x,y,z,m)*my + \
                 kernelyz_sphere(x,y,z,m)*mz
    A *= CM*T2NT
    return A

def sm_bz_sphere(x, y, z, xs, ys, zs, incs, decs):
    '''
    This function calculates the sensibility matrix
    of the z component from the magnetic induction field
    for the foward problem with magnetic spheres.
    
    input
    
    x: float/array - number of points in x
    y: float/array - number of poitns in y
    z: float/array - number of poitns in z
    xs: float/array - source position
    ys: float/array - source position
    zs: float/array - source position
    incs: float - inclination of the sources
    decs: float - declination of the sourcess

    output
    
    A: float/array - sensibility matrix of foward problem
    '''
    
    assert x.size == y.size == z.size, 'The number of points in x, y and z must be equal'
    assert xs.size == ys.size == zs.size, 'The number of points in xs, ys and zs must be equal'
    
    A = np.empty((x.size,xs.size))
    
    mx, my, mz = utils.ang2vec(1.,incs,decs) # sources direction
    R = (3./(4.*np.pi))**(1./3)

    for j, (xf, yf, zf) in enumerate(zip(xs,ys,zs)):
        m = [xf, yf, zf, R]
                
        A[:,j] = kernelxz_sphere(x,y,z,m)*mx + \
                 kernelyz_sphere(x,y,z,m)*my + \
                 kernelzz_sphere(x,y,z,m)*mz
        
    A *= CM*T2NT
    return A

def sm_btb_sphere(x, y, z, xs, ys, zs, incs, decs):
    '''
    This function calculates the sensibility matrix
    of the magnetic induction field amplitude
    for the foward problem with magnetic spheres.
    
    input
    
    x: float/array - number of points in x
    y: float/array - number of poitns in y
    z: float/array - number of poitns in z
    xs: float/array - source position
    ys: float/array - source position
    zs: float/array - source position
    incs: float - inclination of the sources
    decs: float - declination of the sources

    output
    
    A: float/array - sensibility matrix of foward problem
    '''
    
    assert x.size == y.size == z.size, 'The number of points in x, y and z must be equal'
    assert xs.size == ys.size == zs.size, 'The number of points in xs, ys and zs must be equal'
    
    A = np.empty((x.size,xs.size))
    
    mx, my, mz = utils.ang2vec(1.,incs,decs) # sources direction
    R = (3./(4.*np.pi))**(1./3.)
   
    for j, (xf, yf, zf) in enumerate(zip(xs,ys,zs)):
        m = [xf, yf, zf, R]
                
        A[:,j] = (kernelxx_sphere(x,y,z,m)*mx + \
                  kernelxy_sphere(x,y,z,m)*my + \
                  kernelxz_sphere(x,y,z,m)*mz)* \
                 (kernelxx_sphere(x,y,z,m)*mx + \
                  kernelxy_sphere(x,y,z,m)*my + \
                  kernelxz_sphere(x,y,z,m)*mz)+ \
                 (kernelxy_sphere(x,y,z,m)*mx + \
                  kernelyy_sphere(x,y,z,m)*my + \
                  kernelyz_sphere(x,y,z,m)*mz)* \
                 (kernelxy_sphere(x,y,z,m)*mx + \
                  kernelyy_sphere(x,y,z,m)*my + \
                  kernelyz_sphere(x,y,z,m)*mz)+ \
                 (kernelxz_sphere(x,y,z,m)*mx + \
                  kernelyz_sphere(x,y,z,m)*my + \
                  kernelzz_sphere(x,y,z,m)*mz)* \
                 (kernelxz_sphere(x,y,z,m)*mx + \
                  kernelyz_sphere(x,y,z,m)*my + \
                  kernelzz_sphere(x,y,z,m)*mz)
                  
    A *= CM*T2NT*CM*T2NT
    return A