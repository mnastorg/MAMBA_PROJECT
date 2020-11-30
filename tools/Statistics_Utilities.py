##################################################################################################
############### MODULE POUR GERER DES CALCULS DE GEOMETRIES SUR COURBES ##########################
##################################################################################################

import numpy as np

from scipy import integrate
from sklearn.neighbors import NearestNeighbors

from tools import BSplines_Utilities as bs

##################################################################################################
####################################### REPERE DE FRENET #########################################
##################################################################################################

def tangente(DER1):
    """Formule de la tangente de Frenet : T(t) = x'(t)/||x'(t)|| """
    NORM = np.linalg.norm(DER1, axis = 1)
    NORM = NORM[np.newaxis].T
    return DER1/NORM

def binormale(DER1, DER2):
    """ Formule de la binormale de Frenet : B(t) = (x'(t) /crossproduct/ x''(t))/|| x'(t) /crossproduct/ x''(t)|| """
    CROSS = np.cross(DER1, DER2)
    NORM = np.linalg.norm(CROSS, axis = 1)
    NORM = NORM[np.newaxis].T
    return CROSS/NORM

def normale(BINORMALE,TANGENTE):
    """ Formule de la normale de Frenet : N(t) = B(t) /crossproduct/ T(t) """
    CROSS = np.cross(BINORMALE,TANGENTE)
    return CROSS

def frenet_frame(POINTS):
    #NUMBER OF POINTS
    n = len(POINTS)
    t = np.linspace(0, 1, n)
    degree = 3
    knot = bs.Knotvector(POINTS, degree)

    BSPLINE, DER1, DER2, DER3 = bs.BSplines_RoutinePython(POINTS, knot, degree, t, dimension = 3)

    T = tangente(DER1)
    B = binormale(DER1, DER2)
    N = normale(B,T)

    return BSPLINE, T, N, B
###############################################################################################
################################## COURBURE ET TORSION ########################################
###############################################################################################

def courbure(DER1,DER2):
    """ Formule de la courbure : || x'(t) \crossproduct x''(t)|| / || x'(t)||^3 """
    CROSS = np.cross(DER1,DER2)
    NORM_CROSS = np.linalg.norm(CROSS, axis = 1)
    NORM_DER1 = np.linalg.norm(DER1, axis = 1)
    courbure = NORM_CROSS / (NORM_DER1**3)

    return courbure[np.newaxis].T

def torsion(DER1, DER2, DER3):
    """ Formule de la torsion : ((x'(t) /crossproduct/ x''(t)).x'''(t))/||x'(t) /crossproduct/ x''(t)||**2"""
    CROSS = np.cross(DER1,DER2)
    UP = np.zeros(np.shape(DER1)[0])
    for i in range(np.shape(DER1)[0]):
        UP[i] = np.dot(CROSS[i,:],DER3[i,:])
    NORM_CROSS = np.linalg.norm(CROSS, axis = 1)
    torsion = UP / (NORM_CROSS**2)

    return torsion[np.newaxis].T

###############################################################################################
######################################## ARC LENGTH ###########################################
###############################################################################################

def arc_length(tfinal, CTRL):

    degree = 3
    knotvector = bs.Knotvector(CTRL, degree)

    length = integrate.fixed_quad(func_to_integrate, 0.0, tfinal, args = (CTRL, degree, knotvector), n = 10)

    return length[0]

def func_to_integrate(t, CTRL, degree, knotvector):
    DER = bs.BSplines_BasisFunction(CTRL, knotvector, degree, t, 1)
    function = np.sqrt(DER[:,0]**2 + DER[:,1]**2 + DER[:,2]**2)

    return function

def liste_length_point(t, CTRL, degree, knotvector):

    liste_length = []

    for i in range(len(t)):
        liste_length.append(arc_length(t[i],CTRL,degree,knotvector))

    return liste_length

###############################################################################################
######################################## RAYON DU VAISSEAU VIA LEVELSET #######################
###############################################################################################

def rayon_vaisseau(BSPLINES, DIST):
    nbrs = NearestNeighbors(n_neighbors = 1, algorithm = 'auto').fit(DIST[:,0:3])
    distance, indices = nbrs.kneighbors(BSPLINES[:,0:3])
    RADIUS = DIST[indices, 3]

    return RADIUS
