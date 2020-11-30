##################################################################################################
############### MODULE POUR GERER LES CALCULS SUR LES BSPLINES ###################################
##################################################################################################

import numpy as np

from geomdl import BSpline
from geomdl import knotvector
from geomdl import helpers
from geomdl import operations

def BSplines_RoutinePython(CONTROL, KNOT, degree, t, dimension = 2):
    """ Retourne un tableau numpy des coordonnées de la Bspline. Ses
    dérivées 1, 2 et 3. CONTROL est un tableau numpy des coordonnees
    des pts de controles et knot est une liste issu de Knotvector.
    Cette méthode utilise la fonction evaluation de NURBS Python"""

    #Creation de la courbe
    crv = BSpline.Curve()

    #Degre de la courbe
    crv.degree = degree

    #Initialisation des points de controles
    crv.ctrlpts = CONTROL.tolist()

    #Les noeuds
    crv.knotvector = KNOT

    if dimension == 2 :
        BSPLINE = np.zeros((len(t),2))
        DER1 = np.zeros((len(t),2))
        DER2 = np.zeros((len(t),2))
        DER3 = np.zeros((len(t),2))

        for i in range(len(t)) :
            DER = crv.derivatives(t[i], order = 3)
            BSPLINE[i,:] = DER[0]
            DER1[i,:] = DER[1]
            DER2[i,:] = DER[2]
            DER3[i,:] = DER[3]

    if dimension == 3 :
        BSPLINE = np.zeros((len(t),3))
        DER1 = np.zeros((len(t),3))
        DER2 = np.zeros((len(t),3))
        DER3 = np.zeros((len(t),3))

        for i in range(len(t)) :
            DER = crv.derivatives(t[i], order = 3)
            BSPLINE[i,:] = DER[0]
            DER1[i,:] = DER[1]
            DER2[i,:] = DER[2]
            DER3[i,:] = DER[3]

    return BSPLINE, DER1, DER2, DER3

def BSplines_BasisFunction(CONTROL, knot, degree, t, ordre):
    """ Retourne un tableau numpy des coordonnées de la Bspline.
    CONTROL est un tableau numpy des coordonnees des pts de controles
    et knot est une liste issu de Knotvector. Cette méthode utilise
    lae calcul de la matrice des fonctions de base issue de Matrix_Basis_Function"""

    BASIS = Matrix_Basis_Function(degree, knot, np.shape(CONTROL)[0], t, ordre)
    BSPLINE = np.dot(BASIS,CONTROL)

    return BSPLINE

def Knotvector(CONTROL, degree):
    """Génère un vecteur de noeud uniforme en fonction des points
    de controle et du degré d'approximation"""

    #Creation de la courbe
    crv = BSpline.Curve()

    #Degre de la courbe
    crv.degree = degree

    #Initialisation des points de controles
    crv.ctrlpts = CONTROL.tolist()

    #Generation des knots
    crv.knotvector = knotvector.generate(crv.degree, crv.ctrlpts_size)

    return crv.knotvector

def Matrix_Basis_Function(degree, knotvector, nbcontrol, t, ordre):
    """ Génère la matrice des fonctions de base pour le calcul
    de la BSpline en fonction de vecteur de noeud et du nombre de
    points de controle. C'est une matrice de taille (len(t),nbcontrol)."""

    if isinstance(t, int) or isinstance(t, float):
        BASIS = np.zeros((1,nbcontrol))
        for i in range(nbcontrol):
            BASIS[0,i] = Basis_Function_Der(degree, knotvector, i, t, ordre)

    else :
        BASIS = np.zeros((len(t),nbcontrol))
        for i in range(len(t)):
            for j in range(nbcontrol):
                BASIS[i,j] = Basis_Function_Der(degree, knotvector, j, t[i], ordre)

    return BASIS

def Basis_Function_Der(degree, knotvector, i, para, ordre):

    nb_control = len(knotvector) - degree - 1

    if (ordre == 0):
        return helpers.basis_function_one(degree, knotvector, i, para)

    if (ordre == 1):
        if (para == 1.0):
            return -helpers.basis_function_ders_one(degree, knotvector, nb_control -1 - i, 0, ordre)[ordre]
        else :
            return helpers.basis_function_ders_one(degree, knotvector, i, para, ordre)[ordre]

    if (ordre == 2):
        if (para == 1.0):
            return helpers.basis_function_ders_one(degree, knotvector, nb_control - 1 - i, 0, ordre)[ordre]
        else :
            return helpers.basis_function_ders_one(degree, knotvector, i, para, ordre)[ordre]
