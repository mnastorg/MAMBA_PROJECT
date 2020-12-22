################################################################################################################
############################################## PACKAGES IMPORTATION ############################################
################################################################################################################

#SYSTEM PACKAGES
import os
import meshio
import shutil

#MATHS PACKAGES
from math import *
import numpy as np
import scipy.stats as st
from scipy.spatial import procrustes
from scipy import interpolate
from sklearn import mixture
from symfit import parameters, variables, sin, cos, Fit
import matplotlib.pyplot as plt
import matplotlib.colors as col

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
############################################### POD ON REDUCED BASIS ###########################################
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

def Wall_radius_pod(thetas, liste_r_theta, epsilon_wall = 1.e-2, epsilon_radius = 1.e-2, read_file = True):

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

    #MAT_A, MAT_B = Fourier_variations(MAT)
    #PHI_A = Proper_orthogonal_decomposition(MAT_A, 1.e-2, nb_mods = False, plot = False)
    #PHI_B = Proper_orthogonal_decomposition(MAT_B, 1.e-1, nb_mods = False, plot = False)

    print("----> Compute solution's coefficients of both reduced basis")
    COEFFS_WALL = Coefficients_base_pod(MAT, PHI_WALL)
    COEFFS_RAYON = Coefficients_base_pod(RAYON, PHI_RAYON)

    #COEFFS_A = Coefficients_base_pod(MAT_A, PHI_A)
    #COEFFS_B = Coefficients_base_pod(MAT_B, PHI_B)

    return PHI_WALL, PHI_RAYON, COEFFS_WALL, COEFFS_RAYON

def Wall_coeff_evolution_pod(thetas, liste_r_theta, epsilon_wall = 1.e-2, read_file = True):

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

    liste_mat_coeff = Liste_coeff_fourier(MAT)

    liste_pod = []
    liste_pod_coeff = []

    for i in range(len(liste_mat_coeff)):
        print("----> Creation of reduced basis for Fourier coeff {}".format(i))
        print("------> Epsilon Wall : ", epsilon_wall)
        COEFF = liste_mat_coeff[i]
        PHI_COEFF = Proper_orthogonal_decomposition(COEFF, epsilon_wall, nb_mods = False, plot = False)
        liste_pod.append(PHI_COEFF)

        print("----> Compute solution's coefficients of reduced basis {}".format(i))
        COEFF_POD = Coefficients_base_pod(COEFF, PHI_COEFF)
        liste_pod_coeff.append(COEFF_POD)

    return liste_pod, liste_pod_coeff

################################################################################################################
################################################################################################################
################################################################################################################


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
################################################################################################################
################################################################################################################


################################################################################################################
############################################### WALL/RADIUS UTILITIES ##########################################
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

def Liste_coeff_fourier(MAT_WALL):

    liste_mat = []
    t = np.linspace(0,1,1000)

    for i in range(len(MAT_WALL)):
        COEFF = MAT_WALL[i,:]
        TAB = COEFF.reshape(24,1000).T
        liste_mat.append(TAB)
        #plt.plot(t, TAB[:,0], 'b')
        #plt.plot(t, TAB[:,5], 'c')
        #plt.plot(t, TAB[:,10], 'r')
        #plt.show()

    return liste_mat

def Fourier_variations(MAT):

    liste_geo = []
    extract = np.arange(0, np.shape(MAT)[1] + 1000, 1000)

    for i in range(len(extract)-1):
        GEO = MAT[:,extract[i]:extract[i+1]]
        liste_geo.append(GEO)

    liste_a_total = []
    liste_b_total = []

    for geo in liste_geo :
        liste_a = []
        liste_b = []
        for j in range(np.shape(geo)[1]-1):
            S0 = geo[:,j]
            S1 = geo[:,j+1]
            a, b, coeff, p, err = st.linregress(S0,S1)
            liste_a.append(a)
            liste_b.append(b)
        liste_a_total.append(liste_a)
        liste_b_total.append(liste_b)

    return np.vstack(liste_a_total).T, np.vstack(liste_b_total).T

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
    ctr = []
    for i in range(np.shape(TAB)[0]):
        theta = TAB[i,0]
        r = TAB[i, 1]
        rotation = np.asarray([np.cos(theta), np.sin(theta), 0])
        vect = r*rotation
        new_point_plan = COORD_PLAN + vect
        ctr.append(new_point_plan)
        new_point_canon = np.dot(PASSAGE, new_point_plan.T).T
        liste_point.append(new_point_canon)

    """
    CTR = np.vstack(ctr)
    plt.scatter(CTR[:,0], CTR[:,1], label = 'Reconstruction_{}'.format(index))
    #plt.scatter(COUPURE_PLAN[:,0], COUPURE_PLAN[:,1], color = 'red', label = 'Original')
    plt.quiver(COORD_PLAN[0], COORD_PLAN[1], 1, 0, color = 'green')
    plt.quiver(COORD_PLAN[0], COORD_PLAN[1], 0, 1, color = 'blue')
    plt.legend()
    #plt.show()
    """

    return np.vstack(liste_point)

################################################################################################################
################################################################################################################
################################################################################################################


################################################################################################################
################################### TOOLS TO PERFORM OPERATIONS ################################################
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

def Gaussian_liste(COEFFS, nb_gaussian = 1, nb_sample = 1):

    nb_mods = np.shape(COEFFS)[1]
    colors = ['blue', 'red', 'green', 'purple', 'brown', 'pink', 'cyan']

    liste_gaussian = []
    liste_alea = []

    for i in range(nb_mods):

        C = COEFFS[:,i].reshape(-1,1)
        gauss = Gaussian_mixture(C, nb_gaussian)
        liste_gaussian.append(gauss)
        liste_alea.append(gauss.sample(nb_sample)[0][0])

    """
        dmin = np.min(C) - 10
        dmax = np.max(C) + 10
        x = np.linspace(dmin, dmax, 1000)
        mean = (gauss.means_)[0]
        sigma = np.sqrt((gauss.covariances_)[0])
        curve = st.norm.pdf(x, mean, sigma)
        plt.plot(x, curve.ravel(), color = colors[i], label = "coeffs_{}".format(i))
        plt.hist(C, normed = 1, color = colors[i])
        plt.title("Gaussian fitting on coefficients")
        plt.legend()
        #plt.plot(x, curve.ravel(), color = colors[i], label = "coeffs_{}".format(i))
        #plt.hist((gauss.sample(50)[0]).ravel(), bins = 20, color = colors[i], normed = 1)
        #plt.legend()
        #plt.title("Random samples taken in Gaussian models")
    #plt.show()
    """
    return liste_gaussian, np.hstack(liste_alea)

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

def Rotation_section(SECTION, ORIGIN, T, N, B, theta):

    #ON SE MET A L'ORIGINE (0,0,0)
    TRANSLATION = SECTION - ORIGIN
    #MATRICE DE PASSAGE ET CHANGEMENT DE BASE
    PASS = Matrice_de_passage(T, N, B)
    SEC_FRENET = np.dot(PASS.T, TRANSLATION.T).T
    #Z ROTATION (CAR TANGENT EST COORD Z)
    ROT = np.asarray([  [np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
    SEC_ROT = np.dot(ROT, SEC_FRENET.T).T
    #ON SE REPLACE DANS COORD CARTESIENNE + TRANSLATION
    SEC_ROT_CART = np.dot(PASS, SEC_ROT.T).T
    NEW_SECTION = SEC_ROT_CART + ORIGIN

    return NEW_SECTION

################################################################################################################
################################################################################################################
################################################################################################################


################################################################################################################
#################################### RANDOM GEOMETRIES GENERATION FUNCTIONS  ###################################
################################################################################################################

def Generator_wall_radius(PHI_CENTERLINE, PHI_WALL, PHI_RADIUS, COEFFS_CENTERLINE, COEFFS_WALL, COEFFS_RADIUS, dilatation, nb_sections, nb_angles, rotation_section = False):
    """Function to generate random geometries based on the data given by the 3 reduced basis (PHI_CENTERLINE, PHI_WALL, PHI_RADIUS), dilatation from procrustes analysis
    and coefficients from training geometries (COEFFS_CENTERLINE, COEFF_WALL, COEFFS_RADIUS). You can set the number of sections and point for one section (nb_sections, nb_angles).
    To align all sections to original N/B plan (first section), set rotation_section = True."""

    print("------> Number of centerline's points : ", nb_sections)
    t_anevrisme = np.linspace(0, 1, nb_sections)
    print("------> Number of thetas / cut :", nb_angles)
    thetas = np.linspace(0, 2*np.pi, 160)

    ############################################################################
    ############# FITTING GAUSSIANS TO COEFFICIENTS ############################
    ############################################################################

    print("------> Fitting Gaussians to coefficients")
    gauss_centerline, alea_centerline = Gaussian_liste(COEFFS_CENTERLINE, 1)
    print("------> Random numbers from gaussian centerline : ", alea_centerline)
    gauss_dilatation, alea_dilatation = Gaussian_liste(np.asarray(dilatation)[np.newaxis].T, 1)
    print("------> Random numbers from gaussian dilatation : ", alea_dilatation)
    gauss_rayon, alea_rayon = Gaussian_liste(COEFFS_RADIUS, 1)
    print("------> Random numbers from gaussian radius : ", alea_rayon)
    #COMMENT FOLLOWING LINE IF YOU DON'T WANT SAME WALL SHAPE EVERYWHERE
    #DON'T FORGET TO UNCOMMENT LINE IN THE BUILDING SECTION LOOP
    gauss_wall, alea_wall = Gaussian_liste(COEFFS_WALL, 1)
    print("------> Random numbers from gaussian wall : ", alea_wall)


    ############################################################################
    ############# CENTERLINE AND RADIUS DISTRIBUTION ###########################
    ############################################################################

    #### RADIUS DISTRIBUTION ##################################
    # NEED INTERPOLATION TO ADAPT TO NUMBER OF SECTIONS DESIRED
    RADIUS = np.sum(PHI_RADIUS*alea_rayon, axis = 1)
    liste_t = np.linspace(0, 1, np.shape(RADIUS)[0])
    #INTERPOLATION OF RADIUS
    func = interpolate.interp1d(np.asarray(liste_t), RADIUS)
    #plt.plot(t_anevrisme, func(t_anevrisme))
    #plt.show()

    #### CENTERLINE CREATION ##################################
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
    BSPLINE, TAN, NOR, BI = stats.frenet_frame(COORD)
    print("------> Arc Length of the centerline : ", stats.arc_length(1,COORD))

    ############################################################################
    ############# BUILDING SECTIONS ON CENTERLINE ##############################
    ############################################################################

    liste_contour = []

    if rotation_section == True :

        liste_rot = []

        for i in range(nb_sections) :

            if i == 0 :

                #COMPUTE A0 FROM RADIUS DISTRIBUTION INTERPOLATION
                a0 = func(t_anevrisme[i])

                #THE FOLLOWING LINE IS THE ONE TO UNCOMMENT IF YOU WANT
                #RANDOM WALL SHAPE / SECTION
                #gauss_wall, alea_wall = Gaussian_liste(COEFFS_WALL, 1)
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
                liste_rot.append(CONTOUR)

            else :

                #COMPUTE A0 FROM RADIUS DISTRIBUTION INTERPOLATION
                a0 = func(t_anevrisme[i])

                #THE FOLLOWING LINE IS THE ONE TO UNCOMMENT IF YOU WANT
                #RANDOM WALL SHAPE / SECTION
                #gauss_wall, alea_wall = Gaussian_liste(COEFFS_WALL, 1)
                vect = np.sum(PHI_WALL*alea_wall, axis = 1)
                step2 = int(len(vect[1:])/2)
                coeff_a = vect[1:step2+1]
                coeff_b = vect[step2+1:]

                #COMPUTE FUNCTION R(THETA)
                R = Compute_Serie_Fourier(thetas, -a0, coeff_a, coeff_b)
                #CLASSIFY IN TABLE THETA/DISTANCE
                TAB = np.hstack((thetas[np.newaxis].T, R[np.newaxis].T))

                #COORDINATES
                C = BSPLINE[i,:]
                T = TAN[i,:]
                N = NOR[i,:]
                B = BI[i,:]

                #RECONSTRUCTION OF SHAPE OF THE SECTION
                PASSAGE = Matrice_de_passage(T, N, B)
                COORD_PLAN = ((np.dot(PASSAGE.T, C.T)).T)
                CONTOUR = Reconstruction_contour(COORD_PLAN, TAB, PASSAGE)
                liste_contour.append(CONTOUR)

                #NORMAL VECTOR FROM FIRST SECTION
                N0 = NOR[0,:]
                #ANGLE BETWEEN CURRENT NORMAL VECTOR AND FIRST SECTION
                theta = atan2(N[1],N[0]) - atan2(N0[1],N0[0])
                #SECTION ROTATION
                NEW_SEC = Rotation_section(CONTOUR, C, T, N, B, -theta)
                liste_rot.append(NEW_SEC)

        L = np.vstack(liste_contour)
        ANEVRISME = np.vstack((BSPLINE, L))

        L2 = np.vstack(liste_rot)
        ANEVRISME_ROT = np.vstack((BSPLINE, L2))

        return ANEVRISME, ANEVRISME_ROT

    else :

        for i in range(nb_sections) :

            #COMPUTE A0 FROM RADIUS DISTRIBUTION INTERPOLATION
            a0 = func(t_anevrisme[i])

            #THE FOLLOWING LINE IS THE ONE TO UNCOMMENT IF YOU WANT
            #RANDOM WALL SHAPE / SECTION
            #gauss_wall, alea_wall = Gaussian_liste(COEFFS_WALL, 1)

            vect = np.sum(PHI_WALL*alea_wall, axis = 1)
            step2 = int(len(vect[1:])/2)
            coeff_a = vect[1:step2+1]
            coeff_b = vect[step2+1:]

            #COMPUTE FUNCTION R(THETA)
            R = Compute_Serie_Fourier(thetas, -a0, coeff_a, coeff_b)
            #CLASSIFY IN TABLE THETA/DISTANCE
            TAB = np.hstack((thetas[np.newaxis].T, R[np.newaxis].T))

            #COORDINATES AND FRENET FRAME
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

def Generator_coeff_evolution(PHI_CENTERLINE, COEFFS_CENTERLINE, liste_pod, liste_pod_coeff, dilatation, nb_sections, nb_angles, rotation_section = False):
    """Function to generate random geometries based on the data given by the reduced basis (PHI_CENTERLINE + liste_pod + dilatation)
    and coefficients from training geometries (COEFFS_CENTERLINE, liste_pod_coeff). You can set the number of sections and point for one section (nb_sections, nb_angles).
    To align all sections to original N/B plan (first section), set rotation_section = True."""

    print("------> Number of centerline's points : ", nb_sections)
    t_anevrisme = np.linspace(0, 1, nb_sections)
    print("------> Number of thetas / cut :", nb_angles)
    thetas = np.linspace(0, 2*np.pi, 160)

    ############################################################################
    ############# FITTING GAUSSIANS TO COEFFICIENTS ############################
    ############################################################################

    print("------> Fitting Gaussians to coefficients")
    gauss_centerline, alea_centerline = Gaussian_liste(COEFFS_CENTERLINE, 1)
    print("------> Random numbers from gaussian centerline : ", alea_centerline)
    gauss_dilatation, alea_dilatation = Gaussian_liste(np.asarray(dilatation)[np.newaxis].T, 1)
    print("------> Random numbers from gaussian dilatation : ", alea_dilatation)


    ############################################################################
    ############# CENTERLINE ###################################################
    ############################################################################

    #### CENTERLINE CREATION ##################################
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
    BSPLINE, TAN, NOR, BI = stats.frenet_frame(COORD)
    print("------> Arc Length of the centerline : ", stats.arc_length(1,COORD))

    ############################################################################
    ############# FOURIER COEFFICIENT EVOLUTION INTERPOLATION ##################
    ############################################################################

    liste_interpolation = []

    for i in range(len(liste_pod)) :
        #FOR EACH COEFFICIENT, CREATE INTERPOLATION TO BUILD SECTIONS
        PHI_WALL = liste_pod[i]
        COEFF_WALL = liste_pod_coeff[i]
        gauss_wall, alea_wall = Gaussian_liste(COEFF_WALL, 1)
        print("------> Random numbers from gaussian fourier coeff {} : ".format(i), alea_wall)
        WALL = np.sum(PHI_WALL*alea_wall, axis = 1)
        liste_t = np.linspace(0, 1, np.shape(PHI_WALL)[0])
        func = interpolate.interp1d(np.asarray(liste_t), WALL)
        liste_interpolation.append(func)

    ############################################################################
    ############# BUILDING SECTIONS ON CENTERLINE ##############################
    ############################################################################

    liste_contour = []

    if rotation_section == True :

        liste_rot = []

        for i in range(nb_sections) :

            if i == 0 :

                fourier_coeff = []

                for j in range(len(liste_interpolation)) :
                    f = liste_interpolation[j]
                    fourier_coeff.append(f(t_anevrisme[i]))

                step2 = int(len(fourier_coeff[1:])/2)
                a0 = fourier_coeff[0]
                coeff_a = np.asarray(fourier_coeff[1:step2+1])
                coeff_b = np.asarray(fourier_coeff[step2+1:])

                #COMPUTE R(THETA)
                R = Compute_Serie_Fourier(thetas, -a0, -coeff_a, -coeff_b)
                #CLASSIFY IN TAB THETA/DISTANCE
                TAB = np.hstack((thetas[np.newaxis].T, R[np.newaxis].T))

                #COORDINATES
                C = BSPLINE[i,:]
                T = TAN[i,:]
                N = NOR[i,:]
                B = BI[i,:]

                #RECONSTRUCTION OF SHAPE OF THE SECTION
                PASSAGE = Matrice_de_passage(T, N, B)
                COORD_PLAN = ((np.dot(PASSAGE.T, C.T)).T)
                CONTOUR = Reconstruction_contour(COORD_PLAN, TAB, PASSAGE)
                liste_contour.append(CONTOUR)
                liste_rot.append(CONTOUR)

            else :

                fourier_coeff = []

                for j in range(len(liste_interpolation)) :
                    f = liste_interpolation[j]
                    fourier_coeff.append(f(t_anevrisme[i]))

                step2 = int(len(fourier_coeff[1:])/2)
                a0 = fourier_coeff[0]
                coeff_a = np.asarray(fourier_coeff[1:step2+1])
                coeff_b = np.asarray(fourier_coeff[step2+1:])

                #COMPUTE R(THETA)
                R = Compute_Serie_Fourier(thetas, -a0, coeff_a, coeff_b)
                #CLASSIFY IN TAB THETA/DISTANCE
                TAB = np.hstack((thetas[np.newaxis].T, R[np.newaxis].T))

                #COORDINATES
                C = BSPLINE[i,:]
                T = TAN[i,:]
                N = NOR[i,:]
                B = BI[i,:]

                #RECONSTRUCTION OF SHAPE OF THE SECTION + ROTATION
                PASSAGE = Matrice_de_passage(T, N, B)
                COORD_PLAN = ((np.dot(PASSAGE.T, C.T)).T)
                CONTOUR = Reconstruction_contour(COORD_PLAN, TAB, PASSAGE)
                liste_contour.append(CONTOUR)
                N0 = NOR[0,:]
                theta = atan2(N[1],N[0]) - atan2(N0[1],N0[0])
                NEW_SEC = Rotation_section(CONTOUR, C, T, N, B, theta)
                liste_rot.append(NEW_SEC)

        L = np.vstack(liste_contour)
        ANEVRISME = np.vstack((BSPLINE, L))

        L2 = np.vstack(liste_rot)
        ANEVRISME_ROT = np.vstack((BSPLINE, L2))

        return ANEVRISME, ANEVRISME_ROT

    else :

        for i in range(nb_sections) :

            fourier_coeff = []

            for j in range(len(liste_interpolation)) :
                f = liste_interpolation[j]
                fourier_coeff.append(f(t_anevrisme[i]))

            step2 = int(len(fourier_coeff[1:])/2)
            a0 = fourier_coeff[0]
            coeff_a = np.asarray(fourier_coeff[1:step2+1])
            coeff_b = np.asarray(fourier_coeff[step2+1:])

            #COMPUTE R(THETA)
            R = Compute_Serie_Fourier(thetas, -a0, -coeff_a, -coeff_b)
            #CLASSIFY IN TAB THETA/DISTANCE
            TAB = np.hstack((thetas[np.newaxis].T, R[np.newaxis].T))

            #COORDINATES
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

################################################################################################################
################################################################################################################
################################################################################################################
