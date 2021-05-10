import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from fatiando.vis import mpl
from fatiando.gravmag import polyprism
import scipy.stats as sp

def plot_prisms(prisms, scale=1.):
    '''
    Returns a list of ordered vertices to build the model
    on matplotlib 3D

    input

    prisms: list - objects of fatiando.mesher.polyprisms
    scale: float - factor used to scale the coordinate values

    output

    verts: list - ordered vertices
    '''

    assert np.isscalar(scale), 'scale must be a scalar'
    assert scale > 0., 'scale must be positive'

    verts = []
    for o in prisms:
        top = []
        bottom = []
        for x, y in zip(o.x, o.y):
            top.append(scale*np.array([y,x,o.z1]))
            bottom.append(scale*np.array([y,x,o.z2]))
        verts.append(top)
        verts.append(bottom)
        for i in range(o.x.size-1):
            sides = []
            sides.append(scale*np.array([o.y[i], o.x[i], o.z1]))
            sides.append(scale*np.array([o.y[i+1], o.x[i+1], o.z1]))
            sides.append(scale*np.array([o.y[i+1], o.x[i+1], o.z2]))
            sides.append(scale*np.array([o.y[i], o.x[i], o.z2]))
            verts.append(sides)
        sides = []
        sides.append(scale*np.array([o.y[-1], o.x[-1], o.z1]))
        sides.append(scale*np.array([o.y[0], o.x[0], o.z1]))
        sides.append(scale*np.array([o.y[0], o.x[0], o.z2]))
        sides.append(scale*np.array([o.y[-1], o.x[-1], o.z2]))
        verts.append(sides)

    return verts

def plot_simple_model_data(x, y, obs, initial, model, filename):
    '''
    Returns a plot of synthetic total-field anomaly
    data produced by the simple model and the true model
    
    input
    x, y: 1D array - Cartesian coordinates of the upward
                    continued total-field anomaly data
    xa, ya: 1D array - Cartesian coordinates of the observations
    obs: 1D array - synthetic total-field anomaly data
    initial: list - fatiando.mesher.PolygonalPrism
                    of the initial approximate
    model: list - list of fatiando.mesher.PolygonalPrism
                    of the simple model
    filename: string - directory and filename of the figure

    output
    fig: figure - plot
    '''

    plt.figure(figsize=(11,5))

    # sinthetic data
    ax=plt.subplot(1,2,1)
    plt.tricontour(y, x, obs, 20, linewidths=0.5, colors='k')
    plt.tricontourf(y, x, obs, 20,
                    cmap='RdBu_r', vmin=np.min(obs),
                    vmax=-np.min(obs)).ax.tick_params(labelsize=12)
    plt.plot(y, x, 'ko', markersize=.25)
    mpl.polygon(initial, '.-r', xy2ne=True)
    plt.xlabel('$y$(km)', fontsize=14)
    plt.ylabel('$x$(km)', fontsize=14)
    clb = plt.colorbar(pad=0.01, aspect=20, shrink=1)
    clb.ax.set_title('nT', pad=-305)
    mpl.m2km()
    clb.ax.tick_params(labelsize=14)
    plt.text(-6700, 3800, '(a)', fontsize=20)

    verts_true = plot_prisms(model, scale=0.001)
    # true model
    ax = plt.subplot(1,2,2, projection='3d')
    ax.add_collection3d(Poly3DCollection(verts_true, alpha=0.3, 
    facecolor='b', linewidths=0.5, edgecolors='k'))

    ax.set_xlim(-2.5, 2.5, 100)
    ax.set_ylim(-2.5, 2.5, 100)
    ax.set_zlim(2, -0.1, 100)
    ax.tick_params(labelsize=14)
    ax.set_ylabel('y (km)', fontsize= 14)
    ax.set_xlabel('x (km)', fontsize= 14)
    ax.set_zlabel('z (km)', fontsize= 14)
    ax.view_init(10, 50)
    ax.text2D(-0.1, 0.07, '(b)', fontsize=20)

    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')

    return plt.show()

def plot_matrix(z0, intensity, matrix, vmin,
    vmax, solutions, xtitle, ytitle, unity,
    figsize, dpi=300,
    truevalues=[], filename=''):
    '''
    Returns a plot of the goal function values for each inversion
    organized in a matrix
    
    input
    z0: 1D array - range of depth to the top values in meters
    intensity: 1D array - range of total-magnetization
                        intensity values in nT
    matrix: 2D array - values for the goal or misfit function
                    produced by the solutions of the multiple
                    inversions
    vmin: float - minimum value for the colorbar
    vmin: float - maximum value for the colorbar
    solutions: list - list of position on the map of the chosen
                        solutions for the plots [[x1, y1],[x2, y2]]
    xtitle: string - x axis title
    ytitle: string - y axis title
    unity: string - unity of the function
    figsize: tuple - size of the figure
    dpi: integer - resolution of the figure
    truevalues: list - list of position [x, y] on the map of the
                true values for the parameters z0 and intensity
    filename: string - directory and filename of the figure

    output
    fig: figure - plot of the result
    '''
    n = z0.size
    m = intensity.size
    
    fig, ax = fig, ax = plt.subplots(1,1)
    fig.set_size_inches(6,5)
    w = 3
    img = ax.imshow(matrix, vmin=vmin, vmax=vmax, origin='lower',extent=[0,w,0,w])
    img.axes.tick_params(labelsize=14)
    plt.ylabel(ytitle, fontsize=14)
    plt.xlabel(xtitle, fontsize=14)
    if truevalues == []:
        pass
    else:
        plt.plot((2.*truevalues[1]+1.)*w/(2.*m), (2.*truevalues[0]+1.)*w/(2.*n), '^r', markersize=12)
    colors = ['Dw', 'Dm']
    for s, c in zip(solutions, colors):
        plt.plot((2.*s[1]+1.)*w/(2.*m), (2.*s[0]+1.)*w/(2.*n), c, markersize=12)
    x_label_list = []
    y_label_list = []
    for xl, yl in zip(intensity,z0):
        x_label_list.append(str(xl)[:-2])
        y_label_list.append(str(yl)[:-2])
    ax.set_xticks(np.linspace(w/(2.*n), w - w/(2.*n), n))
    ax.set_yticks(np.linspace(w/(2.*m), w - w/(2.*m), m))
    ax.set_xticklabels(x_label_list)
    ax.set_yticklabels(y_label_list)
    # Minor ticks
    ax.set_xticks(np.linspace(0, w, n+1), minor=True)
    ax.set_yticks(np.linspace(0, w, m+1), minor=True)
    ax.grid(which='minor', color='k', linewidth=2)
    clb = plt.colorbar(img, pad=0.01, aspect=20, shrink=1)
    clb.ax.set_title(unity, pad=-288)
    clb.ax.tick_params(labelsize=14)
    if filename == '':
        pass
    else:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    return plt.show()

def plot_complex_model_data(x, y, obs, alt, initial, model,
        figsize, dpi=300, filename=''):
    '''
    Returns a plot of synthetic total-field anomaly
    data produced by the complex model and the true model
    
    input
    x, y: 1D array - Cartesian coordinates of the upward
                    continued total-field anomaly data
    xa, ya: 1D array - Cartesian coordinates of the observations
    obs: 1D array - synthetic total-field anomaly data
    alt: 1D array - geometric heigt of the observations
    initial: list - fatiando.mesher.PolygonalPrism
                    of the initial approximate
    model: list - list of fatiando.mesher.PolygonalPrism
                    of the simple model
    figsize: tuple - size of the figure
    dpi: integer - resolution of the figure
    filename: string - directory and filename of the figure

    output
    fig: figure - plot
    '''

    verts_true = plot_prisms(model, scale=0.001)

    plt.figure(figsize=figsize)

    # sinthetic data
    ax=plt.subplot(2,2,1)
    plt.tricontour(y, x, obs, 20, linewidths=0.5, colors='k')
    plt.tricontourf(y, x, obs, 20,
                    cmap='RdBu_r', vmin=np.min(obs),
                    vmax=-np.min(obs)).ax.tick_params(labelsize=10)
    plt.plot(y, x, 'ko', markersize=.25)
    plt.xlabel('$y$(km)', fontsize=14)
    plt.ylabel('$x$(km)', fontsize=14)
    clb = plt.colorbar(pad=0.01, aspect=20, shrink=1)
    clb.ax.set_title('nT', pad=-315, fontsize=14)
    mpl.polygon(initial, '.-r', xy2ne=True)
    mpl.m2km()
    clb.ax.tick_params(labelsize=13)
    plt.plot(y, x, 'k.', markersize=.25)
    plt.text(np.min(y)-1200, np.max(x), '(a)', fontsize= 20)

    # plot elevation
    ax=plt.subplot(2,2,2)
    plt.tricontourf(y, x, alt, 20,
                    cmap='gray').ax.tick_params(labelsize=10)
    plt.xlabel('$y$(km)', fontsize=14)
    plt.ylabel('$x$(km)', fontsize=14)
    clb = plt.colorbar(pad=0.01, aspect=20, shrink=1)
    clb.ax.set_title('m', pad=-315, fontsize=14)
    mpl.m2km()
    clb.ax.tick_params(labelsize=13)
    plt.plot(y, x, 'ko', markersize=.25)
    plt.text(np.min(y)-1200, np.max(x), '(b)', fontsize= 20)

    # true model
    ax = plt.subplot(2,2,3, projection='3d')
    ax.add_collection3d(Poly3DCollection(verts_true, alpha=1, 
    facecolor='b', linewidths=0.5, edgecolors='k'))

    ax.set_xlim(-2.5, 2.5, 100)
    ax.set_ylim(-2.5, 2.5, 100)
    ax.set_zlim(7, -0.2, 100)
    ax.tick_params(labelsize= 14)
    ax.set_ylabel('y (km)', fontsize= 14, labelpad=10)
    ax.set_xlabel('x (km)', fontsize= 14, labelpad=10)
    ax.set_zlabel('z (km)', fontsize= 14, labelpad=2)
    ax.view_init(10, 55)
    ax.text2D(-0.115, 0.07, '(c)', fontsize= 20)

    # true model
    ax = plt.subplot(2,2,4, projection='3d')
    ax.add_collection3d(Poly3DCollection(verts_true, alpha=1, 
    facecolor='b', linewidths=0.5, edgecolors='k'))

    ax.set_xlim(-2.5, 2.5, 100)
    ax.set_ylim(-2.5, 2.5, 100)
    ax.set_zlim(7, -0.2, 100)
    ax.tick_params(labelsize= 14)
    ax.set_ylabel('y (km)', fontsize= 14, labelpad=10)
    ax.set_xlabel('x (km)', fontsize= 14, labelpad=10)
    ax.set_zlabel('z (km)', fontsize= 14, labelpad=2)
    ax.view_init(20, 145)
    ax.text2D(-0.116, 0.07, '(d)', fontsize= 20)

    plt.tight_layout()

    if filename == '':
        pass
    else:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')

    return plt.show()

def plot_inclined_model_data(x, y, obs, alt, initial, model,
        figsize, dpi=300, filename=''):
    '''
    Returns a plot of synthetic total-field anomaly
    data produced by the inclined model and the true model
    
    input
    x, y: 1D array - Cartesian coordinates of the upward
                    continued total-field anomaly data
    xa, ya: 1D array - Cartesian coordinates of the observations
    obs: 1D array - synthetic total-field anomaly data
    alt: 1D array - geometric heigt of the observations
    initial: list - fatiando.mesher.PolygonalPrism
                    of the initial approximate
    model: list - list of fatiando.mesher.PolygonalPrism
                    of the simple model
    figsize: tuple - size of the figure
    dpi: integer - resolution of the figure
    filename: string - directory and filename of the figure

    output
    fig: figure - plot
    '''

    verts_true = plot_prisms(model, scale=0.001)

    plt.figure(figsize=figsize)

    #===============================================================
    # sinthetic data
    ax=plt.subplot(2,2,1)
    plt.tricontour(y, x, obs, 10, linewidths=0.1, colors='k')
    plt.tricontourf(y, x, obs, 10,
                    cmap='RdBu_r', vmin=np.min(obs),
                    vmax=-np.min(obs)).ax.tick_params(labelsize=14)
    plt.plot(y, x, 'k.', markersize=.1)
    plt.xlabel('$y$(km)', fontsize=14)
    plt.ylabel('$x$(km)', fontsize=14)
    clb = plt.colorbar(pad=0.01, aspect=20, shrink=1)
    clb.ax.set_title('nT', pad=-260, fontsize=14)
    mpl.polygon(initial, '-r', xy2ne=True)
    mpl.m2km()
    clb.ax.tick_params(labelsize=14)
    plt.text(np.min(y)-1000, np.max(x)+1000, '(a)', fontsize=20)

    #==================================================================
    # plot elevation
    ax=plt.subplot(2,2,2)
    plt.tricontourf(y, x, alt, 10,
                    cmap='gray').ax.tick_params(labelsize=14)
    plt.xlabel('$y$(km)', fontsize=14)
    plt.ylabel('$x$(km)', fontsize=14)
    clb = plt.colorbar(pad=0.01, aspect=20, shrink=1)
    clb.ax.set_title('m', pad=-260, fontsize=14)
    mpl.m2km()
    clb.ax.tick_params(labelsize=14)
    plt.plot(y, x, 'k.', markersize=.1)
    plt.text(np.min(y)-1000, np.max(x)+1000, '(b)', fontsize=20)

    #=====================================================================
    # true model
    ax = plt.subplot(2,2,3, projection='3d')
    ax.add_collection3d(Poly3DCollection(verts_true, alpha=0.3, 
    facecolor='b', linewidths=0.5, edgecolors='k'))

    ax.set_ylim(-1.,6., 100)
    ax.set_xlim(-2.5,2., 100)
    ax.set_zlim(3.5, -0.2, 100)
    ax.tick_params(labelsize=14, pad=-2)
    ax.set_ylabel('x (km)', fontsize=14)
    ax.set_xlabel('y (km)', fontsize=14)
    ax.set_zlabel('z (km)', fontsize=14)
    ax.view_init(5, 37)
    ax.text2D(-0.11, 0.07, '(c)', fontsize=20)

    #===================================================================
    # true model
    ax = plt.subplot(2,2,4, projection='3d')
    ax.add_collection3d(Poly3DCollection(verts_true, alpha=0.3, 
    facecolor='b', linewidths=0.5, edgecolors='k'))

    ax.set_ylim(-1.,6., 100)
    ax.set_xlim(-2.5,2., 100)
    ax.set_zlim(3.5, -0.2, 100)
    ax.tick_params(labelsize=14, pad=-2)
    ax.set_ylabel('x (km)', fontsize=14)
    ax.set_xlabel('y (km)', fontsize=14)
    ax.set_zlabel('z (km)', fontsize=14)
    ax.view_init(2, -150)
    ax.text2D(-0.11, 0.07, '(d)', fontsize=20)

    #plt.subplots_adjust(wspace=.5, hspace=.6)

    if filename == '':
        pass
    else:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')

    return plt.show()

def plot_inter_model_data(x, y, z, obs, initial, model,
    inc, dec, figsize, dpi=300, 
    angles=[], area=[], filename=''):
    '''
    Returns a plot of synthetic total-field anomaly
    data produced by the complex model and the true model
    
    input
    x, y, z: 1D array - Cartesian coordinates of the 
                        total-field anomaly data
    xa, ya: 1D array - Cartesian coordinates of the observations
    obs: 1D array - synthetic total-field anomaly data
    alt: 1D array - geometric heigt of the observations
    initial: list - fatiando.mesher.PolygonalPrism
                    of the initial approximate
    insetposition: tuple - position of the inset histogram   
    model: list - list of fatiando.mesher.PolygonalPrism
                    of the simple model
    inc, dec: float - inclination and declination of
                        Earth's main field
    figsize: tuple - size of the figure
    dpi: integer - resolution of the figure
    filename: string - directory and filename of the figure

    output
    fig: figure - plot
    '''
    inter_data = polyprism.tf(x, y, z, [model[-1]], inc, dec)

    if area != []:
        pass
    else:
        area = [np.min(x)/1000., np.max(x)/1000.,
        np.min(y)/1000., np.max(y)/1000.]

    if angles != []:
        pass
    else:
        angles = [10, 50, 10, 50, 20, 160]
    
    V = model[0].x.size

    verts_true = plot_prisms(model, scale=0.001)

    plt.figure(figsize=figsize)

    # sinthetic data
    ax=plt.subplot(2,2,1)
    plt.tricontour(y, x, obs, 20, linewidths=0.5, colors='k')
    plt.tricontourf(y, x, obs, 20,
                    cmap='RdBu_r', vmin=-np.max(obs),
                    vmax=np.max(obs)).ax.tick_params(labelsize=10)
    plt.plot(y, x, 'ko', markersize=.25)
    plt.xlabel('$y$(km)', fontsize=14)
    plt.ylabel('$x$(km)', fontsize=14)
    clb = plt.colorbar(pad=0.01, aspect=20, shrink=1)
    clb.ax.set_title('nT', pad=-315)
    #mpl.polygon(initial, '.-r', xy2ne=True)
    mpl.polygon(model[0], '.-b', xy2ne=True)
    mpl.polygon(model[-1], '.-y', xy2ne=True)
    mpl.m2km()
    clb.ax.tick_params(labelsize=13)
    plt.plot(y, x, 'k.', markersize=.25)
    plt.text(-5600, 3800, '(a)', fontsize= 15)

    # plot interfering data
    ax=plt.subplot(2,2,2)
    plt.tricontour(y, x, inter_data, 20, linewidths=0.5, colors='k')
    plt.tricontourf(y, x, inter_data, 20,
                    cmap='RdBu_r', vmin=-np.max(inter_data),
                    vmax=np.max(inter_data)).ax.tick_params(labelsize=10)
    plt.xlabel('$y$(km)', fontsize=14)
    plt.ylabel('$x$(km)', fontsize=14)
    clb = plt.colorbar(pad=0.01, aspect=20, shrink=1)
    clb.ax.set_title('nT', pad=-315)
    mpl.polygon(model[-1], '.-y', xy2ne=True)
    mpl.m2km()
    clb.ax.tick_params(labelsize=13)
    plt.plot(y, x, 'ko', markersize=.25)
    plt.text(-5600, 3800, '(b)', fontsize= 15)

    # true model
    ax = plt.subplot(2,2,3, projection='3d')
    ax.add_collection3d(Poly3DCollection(verts_true[:-V/2-2], alpha=0.3, 
    facecolor='b', linewidths=0.5, edgecolors='k'))
    ax.add_collection3d(Poly3DCollection(verts_true[-V/2-2:], alpha=1., 
    facecolor='y', linewidths=0.5, edgecolors='k'))

    ax.set_xlim(area[0]/2, area[1]/2, 100)
    ax.set_ylim(area[2]/2, area[3]/2, 100)
    ax.set_zlim(7, -0.2, 100)
    ax.tick_params(labelsize= 10)
    ax.set_ylabel('y (km)', fontsize= 14)
    ax.set_xlabel('x (km)', fontsize= 14)
    ax.set_zlabel('z (km)', fontsize= 14)
    ax.view_init(15, 45)
    ax.text2D(-0.12, 0.07, '(c)', fontsize= 15)

    # true model
    ax = plt.subplot(2,2,4, projection='3d')
    ax.add_collection3d(Poly3DCollection(verts_true[:V+2], alpha=0.3, 
    facecolor='b', linewidths=0.5, edgecolors='k'))
    ax.add_collection3d(Poly3DCollection(verts_true[-V/2-2:], alpha=1., 
    facecolor='y', linewidths=0.5, edgecolors='k'))

    ax.set_xlim(area[0]/4. + 0.5, area[1]/4. +1., 100)
    ax.set_ylim(area[2]/4., area[3]/4., 100)
    ax.set_zlim(1, -0.2, 100)
    ax.tick_params(labelsize= 10)
    ax.set_ylabel('y (km)', fontsize= 14)
    ax.set_xlabel('x (km)', fontsize= 14)
    ax.set_zlabel('z (km)', fontsize= 14)
    ax.view_init(15, 135)
    ax.text2D(-0.12, 0.07, '(d)', fontsize= 15)

    if filename == '':
        pass
    else:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')

    return plt.show()

def plot_solution(xp, yp, zp,
    residuals, solution, initial,
    figsize, dpi=300, insetposition=(0.5, 0.95),
    angles=[], area=[], model=[],
    filename='', inter=False):
    '''
    Returns a plot of the resdiuals, initial approximate
    and two perspective views of the solution for the
    complex model
    
    input
    xp, yp, zp: 1D array - Cartesian coordinates of the residuals
    residuals: 1D array - residuals between observed and predicted data
    solution: list - list of a fatiando.mesher.PolygonalPrism
                    of the estimated model
    initial: list - list of a fatiando.mesher.PolygonalPrism
                    of the initial approximate
    figsize: tuple - size of the figure
    dpi: integer - resolution of the figure
    insetposition: tuple - position of the inset histogram
    angles: list - list of perspective angles of the 3D plots,
                    default: [10, 50, 10, 50, 20, 160]
    area: list - list of minimum and maximum values for the
                    Cartesian coord. of the 3D plots
                    [xmin, xmax, ymin, ymax]
    model: list - list of a fatiando.mesher.PolygonalPrism
                    of the true model or a second solution,
                    default: []
    filename: string - directory and filename of the figure
    inter: boolean - presence of an interfering body

    output
    fig: figure - plot of the result
    '''
   # converting coordinates
    x=xp/1000.
    y=yp/1000.

    verts = plot_prisms(solution, scale=0.001)
    verts_initial = plot_prisms(initial, scale=0.001)

    if area != []:
        pass
    else:
        area = [np.min(x), np.max(x), np.min(y),
        np.max(y)]

    if angles != []:
        pass
    else:
        angles = [10, 50, 10, 50, 20, 160]

    if model != []:
        verts_true = plot_prisms(model, scale=0.001)
        V = model[0].x.size
        if inter == False:
            if model[-1].z2 >= solution[-1].z2:
                zb = model[-1].z2/1000. + 0.5
            else:
                zb = solution[-1].z2/1000. + 0.5
        elif inter == True:
            if model[-2].z2 >= solution[-1].z2:
                zb = model[-2].z2/1000. + 0.5
            else:
                zb = solution[-1].z2/1000. + 0.5
    else:
        zb = solution[-1].z2/1000. + 0.5

    plt.figure(figsize=figsize)

    # residual data and histogram
    ax=plt.subplot(2,2,1)
    plt.tricontourf(y, x, residuals, 20,
                    cmap='RdBu_r', vmin=-np.max(residuals),
                    vmax=np.max(residuals)).ax.tick_params(labelsize=14)
    plt.xlabel('$y$(km)', fontsize=14, labelpad=0)
    plt.ylabel('$x$(km)', fontsize=14, labelpad=0)
    plt.ylim(ymax=np.max(x))
    clb = plt.colorbar(pad=0.01, aspect=20, shrink=1)
    clb.ax.set_title('nT', pad=-265, fontsize=14)
    clb.ax.tick_params(labelsize=14)

    # horizontal projection of the prisms
    for s in solution:
        s.x *= 0.001
        s.y *= 0.001
        s.z1 *= 0.001
        s.z2 *= 0.001
        mpl.polygon(s, fill='k', alpha=0.1, linealpha=0.1, xy2ne=True)

    # histogram inset
    inset = inset_axes(ax, width="40%", height="30%", loc=1, borderpad=0.5)
    mean = np.mean(residuals)
    std = np.std(residuals)
    nbins=30
    n, bins, patches = plt.hist(residuals, bins=nbins, range=(-25,25), density=True, facecolor='blue')
    plt.tick_params(labelsize=14)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    inset.text(
        insetposition[0], insetposition[1],
        '$\mu $ = {:.1f}\n$\sigma $ = {:.1f}'.format(mean, std),
        transform=inset.transAxes, fontsize=10,
        va='top', ha='left', bbox=props
        )
    gauss = sp.norm.pdf(bins, mean, std)
    plt.plot(bins, gauss, 'k--', linewidth=1., label='Gaussian')

    ax.text(np.min(y)-1.2, np.max(x), '(a)', fontsize=20)

    # initial approximate
    ax = plt.subplot(2,2,2, projection='3d')

    # plot sides
    ax.add_collection3d(Poly3DCollection(verts_initial, alpha=0.3, 
     facecolor='r', linewidths=0.5, edgecolors='k'))
    if model == []:
        pass
    elif inter == True:
        ax.add_collection3d(Poly3DCollection(verts_true[:-V/2-2], alpha=0.3, 
        facecolor='b', linewidths=0.5, edgecolors='k'))
        ax.add_collection3d(Poly3DCollection(verts_true[-V/2-2:], alpha=1., 
        facecolor='y', linewidths=0.5, edgecolors='k'))
    else:
        ax.add_collection3d(Poly3DCollection(verts_true, alpha=0.3, 
        facecolor='b', linewidths=0.5, edgecolors='k'))

    ax.set_ylim(area[0], area[1], 100)
    ax.set_xlim(area[2], area[3], 100)
    ax.set_zlim(initial[-1].z2/1000. + 0.2, -0.1, 100)
    ax.tick_params(labelsize=14, pad=0)
    ax.set_ylabel('x (km)', fontsize=14, labelpad=10)
    ax.set_xlabel('y (km)', fontsize=14, labelpad=10)
    ax.set_zlabel('z (km)', fontsize=14, labelpad=2)
    ax.set_yticks(np.arange(area[0], area[1], 2))
    ax.set_xticks(np.arange(area[2], area[3], 2))
    ax.view_init(angles[0], angles[1])
    ax.text2D(-0.1, 0.09, '(b)', fontsize=20)

    # inverse model view 1
    ax = plt.subplot(2,2,3, projection='3d')

    # plot sides
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.3, 
     facecolor='r', linewidths=0.5, edgecolors='k'))
    if model == []:
        pass
    elif inter == True:
        ax.add_collection3d(Poly3DCollection(verts_true[:-V/2-2], alpha=0.3, 
        facecolor='b', linewidths=0.5, edgecolors='k'))
        ax.add_collection3d(Poly3DCollection(verts_true[-V/2-2:], alpha=1.,
        facecolor='y', linewidths=0.5, edgecolors='k'))
    else:
        ax.add_collection3d(Poly3DCollection(verts_true, alpha=0.3, 
        facecolor='b', linewidths=0.5, edgecolors='k'))

    ax.set_ylim(area[0], area[1], 100)
    ax.set_xlim(area[2], area[3], 100)
    ax.set_zlim(zb-0.5, -0.2, 100)
    ax.tick_params(labelsize=14, pad=0)
    ax.set_ylabel('x (km)', fontsize=14, labelpad=10)
    ax.set_xlabel('y (km)', fontsize=14, labelpad=10)
    ax.set_zlabel('z (km)', fontsize=14, labelpad=2)
    ax.set_yticks(np.arange(area[0], area[1], 2))
    ax.set_xticks(np.arange(area[2], area[3], 2))
    ax.view_init(angles[2], angles[3])
    ax.text2D(-0.115, 0.09, '(c)', fontsize=20)

    # inverse model view 2
    ax = plt.subplot(2,2,4, projection='3d')

    # plot sides
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.3, 
     facecolor='r', linewidths=0.5, edgecolors='k'))
    if model == []:
        pass
    elif inter == True:
        ax.add_collection3d(Poly3DCollection(verts_true[:-V/2-2], alpha=0.3, 
        facecolor='b', linewidths=0.5, edgecolors='k'))
        ax.add_collection3d(Poly3DCollection(verts_true[-V/2-2:], alpha=1., 
        facecolor='y', linewidths=0.5, edgecolors='k'))
    else:
        ax.add_collection3d(Poly3DCollection(verts_true, alpha=0.3, 
        facecolor='b', linewidths=0.5, edgecolors='k'))

    ax.set_ylim(area[0], area[1], 100)
    ax.set_xlim(area[2], area[3], 100)
    ax.set_zlim(zb-0.5, -0.2, 100)
    ax.tick_params(labelsize=14)
    ax.set_ylabel('x (km)', fontsize=14, labelpad=10)
    ax.set_xlabel('y (km)', fontsize=14, labelpad=10)
    ax.set_zlabel('z (km)', fontsize=14, labelpad=2)
    ax.set_yticks(np.arange(area[0], area[1], 2))
    ax.set_xticks(np.arange(area[2], area[3], 2))
    ax.view_init(angles[4], angles[5])
    ax.text2D(-0.1, 0.09, '(d)', fontsize=20)

    if filename == '':
        pass
    else:
        plt.savefig(filename, dpi=dpi)
    return plt.show()

def plot_field_solution(xp, yp, zp,
    residuals, solution, initial,
    figsize, dpi=300, insetposition=(0.5, 0.95),
    angles=[], area=[], model=[],
    filename='', inter=False):
    '''
    Returns a plot of the resdiuals, initial approximate
    and two perspective views of the solution for the
    complex model
    
    input
    xp, yp, zp: 1D array - Cartesian coordinates of the residuals
    residuals: 1D array - residuals between observed and predicted data
    solution: list - list of a fatiando.mesher.PolygonalPrism
                    of the estimated model
    initial: list - list of a fatiando.mesher.PolygonalPrism
                    of the initial approximate
    figsize: tuple - size of the figure
    dpi: integer - resolution of the figure
    insetposition: tuple - position of the inset histogram
    angles: list - list of perspective angles of the 3D plots,
                    default: [10, 50, 10, 50, 20, 160]
    area: list - list of minimum and maximum values for the
                    Cartesian coord. of the 3D plots
                    [xmin, xmax, ymin, ymax]
    model: list - list of a fatiando.mesher.PolygonalPrism
                    of the true model or a second solution,
                    default: []
    filename: string - directory and filename of the figure
    inter: boolean - presence of an interfering body

    output
    fig: figure - plot of the result
    '''
   # converting coordinates
    x=xp/1000.
    y=yp/1000.

    verts = plot_prisms(solution, scale=0.001)
    verts_initial = plot_prisms(initial, scale=0.001)

    if area != []:
        pass
    else:
        area = [np.min(x), np.max(x), np.min(y),
        np.max(y)]

    if angles != []:
        pass
    else:
        angles = [10, 50, 10, 50, 20, 160]

    if model != []:
        verts_true = plot_prisms(model, scale=0.001)
        V = model[0].x.size
        if inter == False:
            if model[-1].z2 >= solution[-1].z2:
                zb = model[-1].z2/1000. + 0.5
            else:
                zb = solution[-1].z2/1000. + 0.5
        elif inter == True:
            if model[-2].z2 >= solution[-1].z2:
                zb = model[-2].z2/1000. + 0.5
            else:
                zb = solution[-1].z2/1000. + 0.5
    else:
        zb = solution[-1].z2/1000. + 0.5

    plt.figure(figsize=figsize)

    # residual data and histogram
    ax=plt.subplot(2,2,1)
    plt.tricontourf(y, x, residuals, 20,
                    cmap='RdBu_r', vmin=-np.max(residuals),
                    vmax=np.max(residuals)).ax.tick_params(labelsize=14)
    plt.xlabel('$y$(km)', fontsize=14, labelpad=0)
    plt.ylabel('$x$(km)', fontsize=14, labelpad=0)
    plt.ylim(ymax=np.max(x))
    clb = plt.colorbar(pad=0.01, aspect=20, shrink=1)
    clb.ax.set_title('nT', pad=-265, fontsize=14)
    clb.ax.tick_params(labelsize=14)

    # horizontal projection of the prisms
    for s in solution:
        s.x *= 0.001
        s.y *= 0.001
        s.z1 *= 0.001
        s.z2 *= 0.001
        mpl.polygon(s, fill='k', alpha=0.1, linealpha=0.1, xy2ne=True)

    # histogram inset
    inset = inset_axes(ax, width="40%", height="30%", loc=1, borderpad=0.5)
    mean = np.mean(residuals)
    std = np.std(residuals)
    nbins=30
    n, bins, patches = plt.hist(residuals, bins=nbins, range=(-100,100), density=True, facecolor='blue')
    plt.tick_params(labelsize=14)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    inset.text(
        insetposition[0], insetposition[1],
        '$\mu $ = {:.1f}\n$\sigma $ = {:.1f}'.format(mean, std),
        transform=inset.transAxes, fontsize=10,
        va='top', ha='left', bbox=props
        )
    gauss = sp.norm.pdf(bins, mean, std)
    plt.plot(bins, gauss, 'k--', linewidth=1., label='Gaussian')

    ax.text(np.min(y)-1., np.max(x)+1., '(a)', fontsize=20)

    # initial approximate
    ax = plt.subplot(2,2,2, projection='3d')

    # plot sides
    ax.add_collection3d(Poly3DCollection(verts_initial, alpha=0.3, 
     facecolor='r', linewidths=0.5, edgecolors='k'))
    if model == []:
        pass
    elif inter == True:
        ax.add_collection3d(Poly3DCollection(verts_true[:-V/2-2], alpha=0.3, 
        facecolor='b', linewidths=0.5, edgecolors='k'))
        ax.add_collection3d(Poly3DCollection(verts_true[-V/2-2:], alpha=1., 
        facecolor='y', linewidths=0.5, edgecolors='k'))
    else:
        ax.add_collection3d(Poly3DCollection(verts_true, alpha=0.3, 
        facecolor='b', linewidths=0.5, edgecolors='k'))

    ax.set_ylim(area[0]+3, area[1]+1, 100)
    ax.set_xlim(area[2]+1, area[3]-1, 100)
    ax.set_zlim(initial[-1].z2/1000. + 1, -0.5, 100)
    ax.tick_params(labelsize=14)
    ax.set_ylabel('x (km)', fontsize=14, labelpad=10)
    ax.set_xlabel('y (km)', fontsize=14, labelpad=10)
    ax.set_zlabel('z (km)', fontsize=14, labelpad=2)
    ax.set_yticks(np.arange(area[0]+3, area[1]+1, 3))
    ax.set_xticks(np.arange(area[2]+1, area[3]-1, 3))
    ax.view_init(angles[0], angles[1])
    ax.text2D(-0.1, 0.105, '(b)', fontsize=20)

    # inverse model view 1
    ax = plt.subplot(2,2,3, projection='3d')

    # plot sides
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.3, 
     facecolor='r', linewidths=0.5, edgecolors='k'))
    if model == []:
        pass
    elif inter == True:
        ax.add_collection3d(Poly3DCollection(verts_true[:-V/2-2], alpha=0.3, 
        facecolor='b', linewidths=0.5, edgecolors='k'))
        ax.add_collection3d(Poly3DCollection(verts_true[-V/2-2:], alpha=1.,
        facecolor='y', linewidths=0.5, edgecolors='k'))
    else:
        ax.add_collection3d(Poly3DCollection(verts_true, alpha=0.3, 
        facecolor='b', linewidths=0.5, edgecolors='k'))

    ax.set_ylim(area[0], area[1], 100)
    ax.set_xlim(area[2], area[3], 100)
    ax.set_zlim(zb, -0.1, 100)
    ax.tick_params(labelsize=14)
    ax.set_ylabel('x (km)', fontsize=14, labelpad=10)
    ax.set_xlabel('y (km)', fontsize=14, labelpad=10)
    ax.set_zlabel('z (km)', fontsize=14, labelpad=2)
    ax.set_yticks(np.arange(area[0], area[1], 3))
    ax.set_xticks(np.arange(area[2], area[3], 3))
    ax.view_init(angles[2], angles[3])
    ax.text2D(-0.11, 0.09, '(c)', fontsize=20)

    # inverse model view 2
    ax = plt.subplot(2,2,4, projection='3d')

    # plot sides
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.3, 
     facecolor='r', linewidths=0.5, edgecolors='k'))
    if model == []:
        pass
    elif inter == True:
        ax.add_collection3d(Poly3DCollection(verts_true[:-V/2-2], alpha=0.3, 
        facecolor='b', linewidths=0.5, edgecolors='k'))
        ax.add_collection3d(Poly3DCollection(verts_true[-V/2-2:], alpha=1., 
        facecolor='y', linewidths=0.5, edgecolors='k'))
    else:
        ax.add_collection3d(Poly3DCollection(verts_true, alpha=0.3, 
        facecolor='b', linewidths=0.5, edgecolors='k'))

    ax.set_ylim(area[0], area[1], 100)
    ax.set_xlim(area[2], area[3], 100)
    ax.set_zlim(zb, -0.1, 100)
    ax.tick_params(labelsize=14, pad=5)
    ax.set_ylabel('x (km)', fontsize=14, labelpad=10)
    ax.set_xlabel('y (km)', fontsize=14, labelpad=10)
    ax.set_zlabel('z (km)', fontsize=14, labelpad=2)
    ax.set_yticks(np.arange(area[0], area[1], 3))
    ax.set_xticks(np.arange(area[2], area[3], 3))
    ax.view_init(angles[4], angles[5])
    ax.text2D(-0.1, 0.09, '(d)', fontsize=20)

    if filename == '':
        pass
    else:
        plt.savefig(filename, dpi=dpi)
    return plt.show()

def plot_obs_alt(x, y, obs, alt, topo,
    initial, figsize, dpi=300, filename=''):
    '''
    Returns a plot of upward continued total-field anomaly
    data and the elevation of the observations
    
    input
    x, y: 1D array - Cartesian coordinates of the observations
    obs: 1D array - upward continued total-field anomaly data
    alt: 1D array - geometric heigt of the observations
    topo: 1D array - geometric heigt of the topography
    initial: list - fatiando.mesher.PolygonalPrism
                    of the initial approximate
    figsize: tuple - size of the figure
    dpi: integer - resolution of the figure
    filename: string - directory and filename of the figure

    output
    fig: figure - plot
    '''

    plt.figure(figsize=figsize)

    ax=plt.subplot(2,2,1)
    plt.tricontour(y, x, obs, 30, linewidths=0.4, colors='k')
    plt.tricontourf(y, x, obs, 30, cmap='RdBu_r',
                   vmin=-np.max(obs),
                   vmax=np.max(obs)).ax.tick_params(labelsize=12)
    plt.plot(y, x, 'ko', markersize=.25)
    mpl.polygon(initial, '.-r', xy2ne=True)
    plt.xlabel('$y$(km)', fontsize=14)
    plt.ylabel('$x$(km)', fontsize=14)
    clb = plt.colorbar(pad=0.01, aspect=40, shrink=1)
    clb.ax.tick_params(labelsize=13)
    clb.ax.set_title('nT', pad=-285)
    ax.text(np.min(y)-1000, np.max(x)+300, '(a)', fontsize= 15)
    mpl.m2km()

    ax=plt.subplot(2,2,2)
    plt.tricontourf(y, x, alt, 10, cmap='gray').ax.tick_params(labelsize=12)
    plt.plot(y, x, 'ko', markersize=.25)
    plt.xlabel('$y$(km)', fontsize=14)
    plt.ylabel('$x$(km)', fontsize=14)
    mpl.polygon(initial, '.-r', xy2ne=True)
    clb = plt.colorbar(pad=0.01, aspect=40, shrink=1)
    clb.ax.set_title('m', pad=-285)
    clb.ax.tick_params(labelsize=13)
    ax.text(np.min(y)-1000, np.max(x)+300, '(b)', fontsize= 15)
    mpl.m2km()

    ax=plt.subplot(2,2,3)
    plt.tricontourf(y, x, topo, 10, cmap='terrain_r', vmax=300).ax.tick_params(labelsize=12)
    plt.plot(y, x, 'ko', markersize=.25)
    plt.xlabel('$y$(km)', fontsize=14)
    plt.ylabel('$x$(km)', fontsize=14)
    mpl.polygon(initial, '.-r', xy2ne=True)
    clb = plt.colorbar(pad=0.01, aspect=40, shrink=1)
    clb.ax.set_title('m', pad=-285)
    clb.ax.tick_params(labelsize=13)
    ax.text(np.min(y)-1000, np.max(x)+300, '(c)', fontsize= 15)
    mpl.m2km()

    if filename == '':
        pass
    else:
        plt.savefig(filename, dpi=dpi)
    return plt.show()