#########################################################################################################################
############################################## PACKAGES IMPORTATION #####################################################
#########################################################################################################################

#SYSTEM PACKAGES
import os
import meshio
import shutil

#MATHS PACKAGES
import numpy as np
import scipy.stats
from scipy.spatial import procrustes
from scipy import interpolate
from sklearn import mixture
from symfit import parameters, variables, sin, cos, Fit
import matplotlib.pyplot as plt

#UTILITIES FILES
from tools import Files_Management as gf
from tools import BSplines_Utilities as bs
from tools import Statistics_Utilities as stats
from geometry_utilities import Geometry_Treatment as geo_treat

#RELOAD ALL NEEDED PACKAGES
gf.Reload(gf)
gf.Reload(bs)
gf.Reload(stats)
gf.Reload(geo_treat)

################################################################################################################
############################################### POD ON THE 3 BASIS #############################################
################################################################################################################

def Centerline_pod(liste_control, epsilon = 1.e-2):

    print("----> Procrustes Analysis + Save dilatation coefficient")
    #EXTRACT CONTROL POINTS FROM PROCRSUTES ANALYSIS
    procrust_control, disparity = Procrustes(liste_control)
    #EXTRACT DILATATION (SCALING) FROM PROCRUSTES ANALYSIS
    dilatation = Dilatation(liste_control)
    #RECONSTRUCT NEW CENTERLINES FROM PROCRUSTES CONTROL POINTS
    procrust_bspline = Construction_bsplines(procrust_control, 200, 5)

    print("----> Creation Matrix of coefficients")
    #DEGREE OF POLYNOMIAL APPROXIMATION
    degree = 5
    #CREATE MATRIX ON WHICH TO RUN POD OF POLYNOMIAL COEFFICIENTS
    print("------> Degree polynomial approximation : ", degree)
    MAT = Matrice_coefficients_centerline(procrust_bspline, degree_approx = degree)

    print("----> Creation of Reduced Basis")
    print("------> Epsilon Centerline : ", epsilon)
    #COMPUTE POD ANALYSIS ON COEFFICIENTS MATRIX
    #EPSILON ALLOWS AN AUTOMATIC EXTRACTION OF NUMBER OF POD MODS
    #IF YOU WANT A PRECISE NUMBER OF POD MODS CHANGE nb_mods = False TO THE DESIRED NUMBER
    #IF plot = True IT PLOTS THE EIGENVALUES GRAPHS
    PHI_CENTERLINE = Proper_orthogonal_decomposition(MAT, epsilon, nb_mods = False, plot = False)

    print("----> Compute solution's coefficients of reduced basis")
    #EACH SOLUTION (COLUMN OF MAT) CAN BE WRITTEN AS A LINEAR COMBINATION OF POD MODS
    #FOLLOWING FUNCTION COMPUTE THE LINEAR COMBINATION COEFFICIENTS FOR EACH SOLUTION
    COEFFS_CENTERLINE = Coefficients_base_pod(MAT, PHI_CENTERLINE)

    #COMPUTE THE ONE LEFT OUT TECHNIQUE TO ARGUE OUR WORK PRECISION
    #err_one_left = One_left_out(MAT, epsilon, plot = True)
    #print("Mean of one left out error : ", err_one_left)

    return dilatation, PHI_CENTERLINE, COEFFS_CENTERLINE

def Wall_pod(thetas, liste_r_theta, epsilon_wall = 1.e-2, epsilon_radius = 1.e-2, read_file = True):

    #FIRST TIME YOU USE THIS FUNCTION YOU CAN'T WRITE read_file = True
    #COMPUTING THE MATRICES MAT AND RAYON CAN BE VERY LONG. ONCE COMPUTED
    # FOR THE FIRST TIME IT IS RECOMMANDED TO STORE IT IN YOUR COMPUTER
    #AUTOMATICALLY DONE LINE 82 AND 83. THEN NEXT TIME YOU CAN CALL THEM
    #USING read_file = True AND LINE 87 and 89.

    print("----> Creation Matrix of coefficients")
    if read_file == False :
        #SET FOURIER SERIES ORDER
        f_order = 5
        #COMPUTE MATRICES WALL SHAPE AND RADIUS
        MAT, RAYON = Matrice_coefficients_wall(thetas, liste_r_theta, fourier_order = f_order)
        np.savetxt("files/MATRIX_WALL.csv", MAT, delimiter = ', ')
        np.savetxt("files/MATRIX_RAYON.csv", MAT, delimiter = ', ')
    else :
        print("------> Reading existing file")
        MAT = gf.Read_parametrization("files/MATRIX_WALL.csv")
        RAYON = gf.Read_parametrization("files/MATRIX_RADIUS.csv")


    print("----> Creation of wall reduced basis")
    print("------> Epsilon Wall : ", epsilon_wall)
    PHI_WALL = Proper_orthogonal_decomposition(MAT, epsilon_wall, nb_mods = False, plot = False)
    #err_one_left_wall = One_left_out(MAT, epsilon_wall, plot = True)
    #print("------> Mean of one left out error : ", err_one_left_wall)

    print("----> Creation of radius reduced basis")
    print("------> Epsilon Radius : ", epsilon_radius)
    PHI_RAYON = Proper_orthogonal_decomposition(RAYON, epsilon_radius, nb_mods = False, plot = False)
    #err_one_left_radius= One_left_out(RAYON, epsilon_radius, plot = True)
    #print("------> Mean of one left out error : ", err_one_left_radius)

    print("----> Compute solution's coefficients of both reduced basis")
    COEFFS_WALL = Coefficients_base_pod(MAT, PHI_WALL)
    COEFFS_RAYON = Coefficients_base_pod(RAYON, PHI_RAYON)

    return PHI_WALL, PHI_RAYON, COEFFS_WALL, COEFFS_RAYON

################################################################################################################
############################################### CENTERLINE UTILITIES ###########################################
################################################################################################################

def Procrustes(liste_control):

    REF = liste_control[0]

    result = []
    disparity = []

    for i in  liste_control:
        mtx1, mtx2, disp = procrustes(REF, i)
        result.append(mtx2)
        disparity.append(disp)

    return result, disparity

def Dilatation(liste_control):

    dilatation = []
    for i in range(len(liste_control)):
        scale = np.linalg.norm(liste_control[i] - np.mean(liste_control[i], axis=0))
        dilatation.append(scale)

    return dilatation

def Construction_bsplines(procrust_control, nb_points, degree):

    procrust_bsplines = []

    t = np.linspace(0, 1, nb_points)

    for i in range(len(procrust_control)):
        CONTROL = procrust_control[i]
        KNOT = bs.Knotvector(CONTROL, degree)
        BSPLINES, DER1, DER2, DER3 = bs.BSplines_RoutinePython(CONTROL, KNOT, degree, t, dimension = 3)
        BSPLINES = np.insert(BSPLINES, 3, i, axis = 1)
        procrust_bsplines.append(BSPLINES)

    #gf.Write_csv("PROCRUST_BSPLINES.csv", np.vstack(procrust_bsplines), "x, y, z, num")

    return procrust_bsplines

def Matrice_coefficients_centerline(bsplines_procrust, degree_approx = 5):

    coeff_X = []
    coeff_Y = []
    coeff_Z = []

    t = np.linspace(0, 1, np.shape(bsplines_procrust[0])[0])

    for coord in range(3):

        for i in range(len(bsplines_procrust)) :

            coeffs = np.polyfit(t, bsplines_procrust[i][:,coord], deg = degree_approx)

            if coord == 0 :
                coeff_X.append(coeffs)
            elif coord == 1 :
                coeff_Y.append(coeffs)
            else :
                coeff_Z.append(coeffs)

    liste_concat = []

    for x, y, z in zip(coeff_X, coeff_Y, coeff_Z):
        concat = np.concatenate((x,y,z))
        liste_concat.append(concat)

    MAT = np.vstack(liste_concat).T

    return MAT

################################################################################################################
############################################### WALL UTILITIES #################################################
################################################################################################################

def Matrice_coefficients_wall(thetas, liste_r_theta, fourier_order = 5):

    coeffs_fourier = []
    rayon = []

    cpt1 = 0


    for geo in liste_r_theta :

        print("Géométrie numéro : ", cpt1)
        a0 = []
        cpt2 = 0

        for i in range(np.shape(geo)[0]) :
            print(" ------> Coefficient extraction running : {}%".format(round((cpt2/np.shape(geo)[0])*100),2), end = "\r")

            THETA_R_EXP = np.vstack((thetas, geo[i,:])).T
            COEFFS = Modele_Fourier(THETA_R_EXP, ordre = fourier_order)
            coeffs_fourier.append(COEFFS)
            a0.append(COEFFS[0])
            cpt2 += 1
        rayon.append(np.vstack(a0))
        cpt1 += 1

    MAT_PARAM = np.hstack(coeffs_fourier)
    RAYON = np.hstack(rayon)

    return MAT_PARAM, RAYON

def Fourier_series(x, f, n = 0):
    """Construit la série de Fourier comme modèle de référence pour le fitting."""

    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x) for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))

    return series

def Modele_Fourier(THETA_R_EXP, ordre = 5):

    #INITIALISATION DU MODELE
    x, y = variables('x, y')
    model_dict = {y: Fourier_series(x, f = 1, n = ordre)}

    #SELECTION DES DONNEES POUR FITTING
    xdata = THETA_R_EXP[:,0]
    ydata = THETA_R_EXP[:,1]

    #FITTING VIA LA METHODE DE SYMPI
    fit = Fit(model_dict, x = xdata, y = ydata)
    fit_result = fit.execute()
    coeffs = fit_result.params
    COEFFS = np.asarray(list(coeffs.values()))[np.newaxis].T

    return COEFFS

def Compute_Serie_Fourier(thetas, a0, coeff_a, coeff_b):

    serie = np.zeros(len(thetas))
    for j in range(len(thetas)) :
        sum = a0
        for i in range(len(coeff_a)):
            sum += coeff_a[i]*np.cos((i+1)*thetas[j]) + coeff_b[i]*np.sin((i+1)*thetas[j])
        serie[j] = sum
    return serie

def Matrice_de_passage(TAN, NOR, BI):
    """Ressort la matrice de passage de la base canonique vers (NOR, BI, TAN). Le repère
    (NOR, BI, TAN) étant orthonormée l'inverse de la matrice de passage est sa transposée."""
    MAT_PASSAGE = np.zeros((3,3))
    MAT_PASSAGE[:,0] = NOR
    MAT_PASSAGE[:,1] = BI
    MAT_PASSAGE[:,2] = TAN

    return MAT_PASSAGE

def Reconstruction_contour(COORD_PLAN, TAB, PASSAGE):

    liste_point = []
    for i in range(np.shape(TAB)[0]):
        theta = TAB[i,0]
        r = TAB[i, 1]
        rotation = np.asarray([np.cos(theta), np.sin(theta), 0])
        vect = r*rotation
        new_point_plan = COORD_PLAN + vect
        new_point_canon = np.dot(PASSAGE, new_point_plan.T).T
        liste_point.append(new_point_canon)

    return np.vstack(liste_point)

################################################################################################################
############################################### MISCELLANOUS UTILITIES #########################################
################################################################################################################

def Proper_orthogonal_decomposition(MAT, tol, nb_mods = False, plot = False):

    ligne, colonne = np.shape(MAT)

    #CLASSICAL METHOD !

    if ligne <= colonne :

        print("------> POD Classical method")

        PROJ_CLASS = np.dot(MAT,MAT.T)
        U, LAMBDA2, UT = np.linalg.svd(PROJ_CLASS, full_matrices = False)

        if nb_mods != False :
            PHI = U[: , :nb_mods]
            ric = np.cumsum(LAMBDA2)/np.sum(LAMBDA2)
            print("--------> Number of POD mods : ", np.shape(PHI)[1])
            print("--------> Percentage of captured energy : {} %".format(ric[nb_mods-1]*100))
            print("--------> Mean of projection relative error", Erreur_proj(MAT, PHI, plot = False))

        else :
            ric = np.cumsum(LAMBDA2)/np.sum(LAMBDA2)
            val = list(filter(lambda i : i > 1-tol, ric))[0]
            M = list(ric).index(val)
            PHI = U[:,:M+1]
            print("--------> Number of POD mods : ", np.shape(PHI)[1])
            print("--------> Percentage of captured energy : {} %".format(ric[M]*100))
            print("--------> Mean of projection relative error", Erreur_proj(MAT, PHI, plot = False))

    #SNAPSHOTS METHOD !
    else :

        print("------> POD Snapshots method")
        PROJ_SNAP = np.dot(MAT.T, MAT)
        V, LAMBDA2, VT = np.linalg.svd(PROJ_SNAP, full_matrices = False)

        if nb_mods != False :
            PSY = V[:,:nb_mods]
            PHI = np.dot(np.dot(MAT, PSY) , np.diag(1./np.sqrt(LAMBDA2[:nb_mods])))
            ric = np.cumsum(LAMBDA2)/np.sum(LAMBDA2)
            print("--------> Number of POD mods : ", np.shape(PHI)[1])
            print("--------> Percentage of captured energy : {} %".format(ric[nb_mods-1]*100))
            print("--------> Mean of projection relative error", Erreur_proj(MAT, PHI, plot = False))

        else :
            ric = np.cumsum(LAMBDA2)/np.sum(LAMBDA2)
            val = list(filter(lambda i : i > 1-tol, ric))[0]
            M = list(ric).index(val)
            PSY = V[:,:M+1]
            PHI = np.dot(np.dot(MAT, PSY) , np.diag(1./np.sqrt(LAMBDA2[:M+1])))
            print("--------> Number of POD mods : ", np.shape(PHI)[1])
            print("--------> Percentage of captured energy : {} %".format(ric[M]*100))
            print("--------> Mean of projection relative error", Erreur_proj(MAT, PHI, plot = False))

    if plot != False :
        plt.figure()
        plt.plot(np.arange(0,len(LAMBDA2)), LAMBDA2, '-o')
        plt.xlabel("n")
        plt.ylabel("Eigenvalues")
        plt.yscale('log')
        #plt.title(plot)
        plt.show()

    return PHI

def Erreur_proj(MAT, PHI, plot = True):

    PROJ = np.dot(PHI,PHI.T)

    erreur_relative = (np.linalg.norm(np.dot(PROJ,MAT) - MAT, axis=0)**2) / (np.linalg.norm(MAT, axis=0))**2

    erreur_proj_moy = (np.linalg.norm(np.dot(PROJ,MAT) - MAT)**2) / (np.linalg.norm(MAT)**2)

    if plot == True :
        plt.figure()
        plt.plot(np.arange(0,np.shape(MAT)[1]), erreur_relative, '-o')
        plt.xlabel('Geometry')
        plt.ylabel('Projection relative error')
        plt.title("Projection relative error for each solution")
        plt.show()

    return erreur_proj_moy

def Coefficients_base_pod(MAT, PHI):

    COEFFS = np.dot(PHI.T, MAT).T

    return COEFFS

def Gaussian_mixture(COEFFS, nb_gaussian):

    clf = mixture.GaussianMixture(n_components = nb_gaussian, covariance_type = 'full')
    clf.fit(COEFFS)

    return clf

def One_left_out(MAT, epsilon, plot = False):

    nb_sol = np.shape(MAT)[1]
    liste_erreur = []

    for i in range(nb_sol):
        SOL = (MAT[:,i])[np.newaxis].T
        NEW_MAT = np.delete(MAT, i, axis = 1)
        PHI = Proper_orthogonal_decomposition(NEW_MAT, epsilon, nb_mods = False, plot = False)
        PROJ = np.dot(PHI, PHI.T)
        err = (np.linalg.norm(np.dot(PROJ,SOL) - SOL)**2) / (np.linalg.norm(SOL)**2)
        liste_erreur.append(err)

    if plot != False :
        plt.figure()
        plt.plot(np.arange(0,nb_sol), liste_erreur, '-o')
        plt.xlabel("Solution out")
        plt.ylabel("Relative projection error")
        plt.yscale('log')
        plt.title("Relative projection error of a external solution")
        plt.show()

    return np.mean(liste_erreur)

################################################################################################################
#################################### FUNCTION TO GENERATE RANDOM GEOMETRIES ####################################
################################################################################################################

def Aneurysm_generator(PHI_CENTERLINE, PHI_WALL, PHI_RAYON, COEFFS_CENTERLINE, COEFFS_WALL, COEFFS_RAYON, dilatation, nb_coupures, nb_angles):

    print("------> Fitting N-dimensional Gaussian to the coefficients")
    gauss_centerline = Gaussian_mixture(COEFFS_CENTERLINE, 1)
    gauss_dilatation = Gaussian_mixture(np.asarray(dilatation)[np.newaxis].T, 1)
    gauss_wall = Gaussian_mixture(COEFFS_WALL, 1)
    gauss_rayon = Gaussian_mixture(COEFFS_RAYON, 1)

    print("------> Number of centerline's points : ", nb_coupures)
    t_anevrisme = np.linspace(0, 1, nb_coupures)

    print("------> Number of thetas / cut :", nb_angles)
    thetas = np.linspace(0, 2*np.pi, 160)

    # ON S'OCCUPE DE LA CENTERLINE ET DE LA REPARTITION DES RAYONS
    alea_centerline = gauss_centerline.sample(1)[0]
    print("------> Random numbers from gaussian centerline : ", alea_centerline)
    alea_dilatation = gauss_dilatation.sample(1)[0]
    print("------> Random numbers from gaussian dilatation : ", alea_dilatation)
    alea_rayon = gauss_rayon.sample(1)[0]
    print("------> Random numbers from gaussian radius : ", alea_rayon)
    #alea_wall = gauss_wall.sample(1)[0]
    #print("------> Random numbers from gaussian wall : ", alea_wall)

    #RANDOM LINEAR COMBINATION FOR RADIUS DISTRIBUTION
    RAYON = np.sum(PHI_RAYON*alea_rayon, axis = 1)
    liste_t = np.linspace(0, 1, np.shape(RAYON)[0])
    #INTERPOLATION OF RADIUS
    func = interpolate.interp1d(np.asarray(liste_t), RAYON)

    #CENTERLINE CREATION
    CENTERLINE = np.sum(PHI_CENTERLINE*alea_centerline, axis = 1)
    step = int(len(CENTERLINE)/3)
    coeffs_x = CENTERLINE[0:step]
    coeffs_y = CENTERLINE[step:2*step]
    coeffs_z = CENTERLINE[2*step:]
    px = np.poly1d(coeffs_x)
    py = np.poly1d(coeffs_y)
    pz = np.poly1d(coeffs_z)
    der1x = np.polyder(px, m = 1)
    der1y = np.polyder(py, m = 1)
    der1z = np.polyder(pz, m = 1)
    der2x = np.polyder(px, m = 2)
    der2y = np.polyder(py, m = 2)
    der2z = np.polyder(pz, m = 2)

    COORD = np.zeros((len(t_anevrisme),3))
    COORD[:,0] = px(t_anevrisme)
    COORD[:,1] = py(t_anevrisme)
    COORD[:,2] = pz(t_anevrisme)
    COORD *= alea_dilatation
    print("------> Arc Length of the centerline : ", stats.arc_length(1,COORD))

    BSPLINE, TAN, NOR, BI = stats.frenet_frame(COORD)

    #SECTIONS CREATION
    liste_contour = []

    for i in range(nb_coupures) :

        #COMPUTE A0 FROM RADIUS DISTRIBUTION INTERPOLATION
        a0 = func(t_anevrisme[i])

        #RANDOM WALL SHAPE
        alea_wall = gauss_wall.sample(1)[0]
        vect = np.sum(PHI_WALL*alea_wall, axis = 1)
        step2 = int(len(vect[1:])/2)

        coeff_a = vect[1:step2+1]
        coeff_b = vect[step2+1:]

        R = Compute_Serie_Fourier(thetas, -a0, coeff_a, coeff_b)

        TAB = np.hstack((thetas[np.newaxis].T, R[np.newaxis].T))

        #COORD
        C = BSPLINE[i,:]
        T = TAN[i,:]
        N = NOR[i,:]
        B = BI[i,:]
        #RECONSTRUCTION OF SHAPE OF THE SECTION
        PASSAGE = Matrice_de_passage(T, N, B)
        COORD_PLAN = ((np.dot(PASSAGE.T, C.T)).T)
        CONTOUR = Reconstruction_contour(COORD_PLAN, TAB, PASSAGE)
        liste_contour.append(CONTOUR)

    L = np.vstack(liste_contour)
    ANEVRISME = np.vstack((BSPLINE, L))

    return ANEVRISME
