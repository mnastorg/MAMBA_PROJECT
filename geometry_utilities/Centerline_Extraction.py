import numpy as np
from math import *
import time

from scipy.spatial import distance
from scipy import optimize
from scipy import linalg

from tools import Files_Management as gf
from tools import BSplines_Utilities as bs

gf.Reload(gf)
gf.Reload(bs)

###########################################################################################################################################
###################################################### MAIN CREATION CENTERLINE ###########################################################
###########################################################################################################################################

def Extraction(PCL, degree = 5, max_iter = 100, tol = 1.e-8, rig = 0):

    print(" ---> Initial construction of the Bspline curve")
    CONTROL = BSpline_Initiale(PCL)
    nbcontrol = len(CONTROL)
    t = np.linspace(0, 1, 200)
    print(" ------> Number of point in centerline : ", len(t))
    print(" ------> Degree of the BSpline : ", degree)
    KNOT = bs.Knotvector(CONTROL, degree)
    BSPLINE, DER1, DER2, DER3 = bs.BSplines_RoutinePython(CONTROL, KNOT, degree, t, dimension = 3)
    BASIS_DER2 = bs.Matrix_Basis_Function(degree, KNOT, nbcontrol, t, 2)

    print(" ---> Start of the BSpline fitting inside the Point Cloud")
    print(" ------> Max_iterations = ", max_iter)
    print(" ------> error_tolerance = ", tol)
    print(" ------> rigidity = ", rig)

    tab_it = []
    tab_err = [0,1]
    it = 0
    start_time = time.time()

    while it < max_iter and np.abs(tab_err[-1] - tab_err[-2]) > tol :

        #FITTING SIMPLE
        NEW_CONTROL, NEW_BSPLINES, erreur = Fitting(PCL, CONTROL, BSPLINE, KNOT, BASIS_DER2, degree, t, rig)
        tab_err.append(erreur)

        #MISE A JOUR
        CONTROL = NEW_CONTROL
        BSPLINE = NEW_BSPLINES

        #INCREMENTATION
        it += 1

    print(" ------> Final number of iterations = ", it)
    print(" ------> Final error = ", erreur)

    CENTERLINE = BSPLINE

    end_time = time.time()
    print(" ------> Total time of fitting : ", round(end_time - start_time, 2))

    return CENTERLINE

###########################################################################################################################################
#################################################### UTILITES FOR FITTING BSPLINES ########################################################
###########################################################################################################################################

def Fitting(PCL, CONTROL, BSPLINES, KNOT, BASIS_DER2, degree, t, rig):

    ########################################################################################################################################
    ############ FITTING EN 4 PARTIES ######################################################################################################
    ########################################################################################################################################

    ########################################################################################################################################
    ############ VARIABES PRELIMINAIRES ####################################################################################################
    ########################################################################################################################################

    #CALCUL DU NOMBRE TOTAL DE POINTS DE CONTROL
    nb_control  = np.shape(CONTROL)[0]
    controle = Matrix_to_vector(CONTROL, dimension = 3)

    ########################################################################################################################################
    ########### ETAPE 1 - ON CALCULE LA MATRICE DE FOOT POINT ET LA MATRICE BASE ASSOCIEE ##################################################
    ########################################################################################################################################

    #CALCUL VECTEUR INDICE DES FOOTPOINTS
    index_foot_point = Foot_Point(PCL, BSPLINES)
    #ON CREE LA MATRICE DES POINTS DE BSPLINES CORRESPONDANT A CELUI DU NUAGE
    BSPLINES_FOOT_POINT = BSPLINES[index_foot_point]
    #CALCUL DE LA MATRICE DE BASE DES FOOTPOINTS
    FP_BASIS, FP_BASIS_DER1, FP_BASIS_DER2 = Matrix_Basis_Foot_Point(BSPLINES_FOOT_POINT, CONTROL, BSPLINES, KNOT, degree, t)

    ########################################################################################################################################
    ########### ETAPE 2 - CALCUL DU 2ND MEMBRE ET DE LA MATRICE BLOCK ######################################################################
    ########################################################################################################################################

    #CALCUL DE LA MATRICE BLOCK A
    BIG_BASIS = linalg.block_diag(FP_BASIS, FP_BASIS, FP_BASIS)
    #CALCUL DU SECOND MEMBRE b
    SND_OBJ = Matrix_to_vector(PCL - BSPLINES_FOOT_POINT, dimension = 3)

    #CALCUL DE LA MATRICE BLOCK B
    BIG_BASIS_DER2 = linalg.block_diag(linalg.block_diag(FP_BASIS_DER2), linalg.block_diag(FP_BASIS_DER2), linalg.block_diag(FP_BASIS_DER2))
    #CALCUL DU SECOND MEMBRE c
    SND_REG = np.dot(BIG_BASIS_DER2, controle)

    A = np.concatenate((BIG_BASIS, np.sqrt(rig)*BIG_BASIS_DER2))
    b = np.concatenate((SND_OBJ, -np.sqrt(rig)*SND_REG))

    #JUSTE POUR S'ASSURER QUE LE RANG DE A EST MAXIMAL
    rang = np.linalg.matrix_rank(A)
    #print("Le rang de A est-il maximal ? : ", rang == np.shape(A)[1])

    ########################################################################################################################################
    ########### ETAPE 3 - ALGORITHME DE MINIMISATION #######################################################################################
    ########################################################################################################################################

    if (rang == np.shape(A)[1]):

        AtA = np.dot(A.T, A)
        Atb = np.dot(A.T, b)

        D = np.linalg.solve(AtA, Atb)
        e = np.linalg.norm(np.dot(A,D) - b)**2

        NEW_CONTROL, NEW_BSPLINES = BSplines_Update_3D(np.reshape(D, np.shape(D)[0]), CONTROL, KNOT, degree, t)
        nb_points = np.shape(NEW_BSPLINES)[0]
        erreur  = e/(3*nb_points)

    else :

        D0 = np.zeros(3*nb_control)
        D = optimize.minimize(ToMinimize_3D, D0, args = (A,b), method = 'Newton-CG', jac = Gradient, hess = Hessienne)

        NEW_CONTROL, NEW_BSPLINES = BSplines_Update_3D(D.x, CONTROL, KNOT, degree, t)
        nb_points = np.shape(NEW_BSPLINES)[0]
        erreur  = D.fun/(3*nb_points)

    return NEW_CONTROL, NEW_BSPLINES, erreur

def Matrix_to_vector(MATRIX, dimension = 3):
    """Transforme un tableau [x,y,z] en un vecteur de taille
    3x nblignes du tableau et prenant d'abord x puis y puis z"""
    return np.reshape(MATRIX, dimension*np.shape(MATRIX)[0], order = 'F')[np.newaxis].T

def Vector_to_matrix(vector, dimension = 3):
    return np.reshape(vector, (int(len(vector)/dimension) , dimension), order = 'F')

def Foot_Point(PCL, BSPLINES):
    """ Retourne un vecteur d'indices correspondants aux foot points.
    Plus précisément, on associe à tout point du nuage son point son
    équivalent dans la Bsplines."""

    DIST = distance.cdist(PCL, BSPLINES)
    minimum = np.argmin(DIST, axis = 1)

    return minimum

def Matrix_Basis_Foot_Point(BSPLINES_FOOT_POINT, CONTROL, BSPLINES, KNOT, degree, t):
    """ Fonction retournant la matrice de base associé au Foot_Point."""

    nb_control = np.shape(CONTROL)[0]

    nb_foot_point = np.shape(BSPLINES_FOOT_POINT)[0]

    BASIS = np.zeros((nb_foot_point, nb_control))
    BASIS_DER1 = 0*BASIS
    BASIS_DER2 = 0*BASIS

    #ON PARCOURT LES FOOTPOINTS
    for i in range(nb_foot_point):
        #ELEMENT A CHERCHER DANS LA MATRICE DES BSPLINES
        search = BSPLINES_FOOT_POINT[i,:]
        #ON CHERCHE L'INDICE DANS BSPLINES DE search. CELA NOUS DONNERA UNE
        #INDICATION SUR LE 't' CORRESPONDANT DANS NEW_BSPLINES
        index = int( (np.where(np.all(BSPLINES == search, axis = 1)))[0] )

        for j in range(nb_control):
            BASIS[i,j] = bs.Basis_Function_Der(degree, KNOT, j, t[index], 0)
            BASIS_DER1[i,j] = bs.Basis_Function_Der(degree, KNOT, j, t[index], 1)
            BASIS_DER2[i,j] = bs.Basis_Function_Der(degree, KNOT, j, t[index], 2)

    return BASIS, BASIS_DER1, BASIS_DER2

def ToMinimize_3D(D, A, b):

    f_obj = 0.5*np.linalg.norm( np.dot(A , D[np.newaxis].T) - b )**2

    return f_obj

def Gradient(D, A, b):

    GRAD = np.dot(A.T, np.dot(A , D[np.newaxis].T) - b)

    return np.reshape(GRAD, np.shape(GRAD)[0])

def Hessienne(D, A, b):

    return np.dot(A.T, A)

def BSplines_Update_3D(D, CONTROL, KNOT, degree, t):
    """ Ressort les nouveaux points de controles ainsi que les nouvelles bsplines apres
    Update par le vecteur D"""

    nb_control = np.shape(CONTROL)[0]

    DD = Vector_to_matrix(D, dimension = 3)

    #ON CREE LA NOUVELLE BSPLINES
    NEW_CONTROL = CONTROL + DD
    NEW_CONTROL[0,:] = CONTROL[0,:]
    NEW_CONTROL[-1,:] = CONTROL[-1,:]

    NEW_BSPLINES, DER1, DER2, DER3 = bs.BSplines_RoutinePython(NEW_CONTROL, KNOT, degree, t, dimension = 3)

    return NEW_CONTROL, NEW_BSPLINES

def BSpline_Initiale(PCL):

    DIST = distance.cdist(PCL,PCL)
    index = (np.argwhere(DIST == np.max(DIST)))[0,:]
    A = PCL[index[0],:]
    B = PCL[index[1],:]

    t = np.linspace(0, 1, 10)
    CONTROL = np.zeros((len(t),3))
    CONTROL[:,0] = (1-t)*A[0]+t*B[0]
    CONTROL[:,1] = (1-t)*A[1]+t*B[1]
    CONTROL[:,2] = (1-t)*A[2]+t*B[2]

    return CONTROL
