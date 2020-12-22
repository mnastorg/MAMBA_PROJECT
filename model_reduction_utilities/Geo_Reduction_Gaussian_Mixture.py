#SYSTEM PACKAGES
import shutil
import os
import time

#MATHS PACKAGES
import numpy as np
from math import *

# import tools for plotting surfaces
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate

# import tools for generate new surfaces
from sklearn.decomposition import PCA
from sklearn.mixture import GMM

from tools import Files_Management as gf
from tools import Statistics_Utilities as stats
from model_reduction_utilities import Geometric_Model_Reduction as gmr


def Main_generative_algorithm(path_patients_data) :

    print("Start Generation Algorithm \n")

    start_time = time.time()

    #READ THE FOLDER
    folder = sorted([f for f in os.listdir(path_patients_data) if not f.startswith('.')], key = str.lower)
    print("Number of patients data files in folder : {} \n".format(len(folder)))

    # Name of folder were generative surfaces are going to be stored
    path_datas = "results/gen_surfaces/training_set"
    path_datas_random = "results/gen_surfaces/predict_set"

    print(" -> Creation of folder named {} to store all generative surfaces", )
    dir1 = path_datas
    if os.path.exists(dir1):
        shutil.rmtree(dir1)
    os.makedirs(dir1)
    dir2 = path_datas_random
    if os.path.exists(dir2):
        shutil.rmtree(dir2)
    os.makedirs(dir2)

    # creating surfaces training set
    training_set = []
    liste_control = []

    #READ EACH FILE IN FOLDER
    for file in folder :

        print("Creating Generative Surface from patient {} surface ".format(file))

        # name of the .csv file that contains the radii
        surf_file_name = path_patients_data + '/' + file + '/' + file + '_parametrization.csv'

        # loading .csv file (surf_data is of type 'float64')
        #print("surf_file_name: ", surf_file_name, "\n")
        PARA = gf.Read_parametrization(surf_file_name)

        CENTERLINE = PARA[:,0:3]
        extract = np.linspace(0, np.shape(CENTERLINE)[0]-1, 10, dtype = 'int')
        CONTROL = CENTERLINE[extract,:]
        liste_control.append(CONTROL)

        RADIUS = PARA[:,12:]
        n_point = np.shape(RADIUS)[0]
        n_radius = np.shape(RADIUS)[1]

        # preparing mesh for plotting
        nb_centerline_points = np.linspace(0, n_point, n_point, endpoint = True, dtype = int)
        nb_thetas = np.linspace(0, n_radius, n_radius, endpoint = True, dtype = int)
        X, Y = np.meshgrid(nb_centerline_points, nb_thetas)
        Z = RADIUS

        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.plot_surface(X, Y, Z.T, cmap='ocean') # Remember:  rows, cols = Z.shape
        #plt.show()

        # surface plot file name:
        surf_plot_name = path_datas + '/' + file + '_surface.png'
        plt.savefig(surf_plot_name)
        # vista 'dall'alto' (piano X-Y)
        #ax.view_init(90, 90)
        # surface plot seen from above file name:
        #surf_plot_name = surf_folder_name + '/' + name_file + '_surface_XY.png'
        #plt.savefig(surf_plot_name)

        # add surface file to list for generation algorithm
        training_set.append(RADIUS.ravel())

    ######## CENTERLINE ##############
    print("Procrustes Analysis + Save dilatation coefficient + Creation Matrix of coefficients")
    #EXTRACT CONTROL POINTS FROM PROCRSUTES ANALYSIS
    procrust_control, disparity = gmr.Procrustes(liste_control)
    #EXTRACT DILATATION (SCALING) FROM PROCRUSTES ANALYSIS
    dilatation = gmr.Dilatation(liste_control)
    DILATATION = np.asarray(dilatation).reshape(-1,1)
    print("Size of dilatation training set : ", np.shape(DILATATION))
    #RECONSTRUCT NEW CENTERLINES FROM PROCRUSTES CONTROL POINTS
    procrust_bspline = gmr.Construction_bsplines(procrust_control, 200, 5)
    for i in range(len(procrust_bspline)):
        SPLINE = procrust_bspline[i]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(SPLINE[:,0], SPLINE[:,1], SPLINE[:,2]) # Remember:  rows, cols = Z.shape
        spline_plot_name = path_datas + '/' + str(i) + '_spline.png'
        plt.savefig(spline_plot_name)

    #DEGREE OF POLYNOMIAL APPROXIMATION
    degree = 5
    #CREATE MATRIX ON WHICH TO RUN POD OF POLYNOMIAL COEFFICIENTS
    print("Degree polynomial approximation : ", degree)
    TRAIN_COEFF = gmr.Matrice_coefficients_centerline(procrust_bspline, degree_approx = degree)
    print("Size of coefficient training set ", np.shape(TRAIN_COEFF))

    ######### RADIUS ##################
    TRAIN_RADIUS = np.vstack(training_set)
    print("Size of the radius training set : ", np.shape(TRAIN_RADIUS))

    TRAIN = np.hstack((DILATATION, TRAIN_COEFF.T, TRAIN_RADIUS))
    print("Size of the full training set : ", np.shape(TRAIN))

    ##### DIMENSIONALITY REDUCTION ##########
    pca = PCA(0.99999, whiten = True, svd_solver = 'full')
    REDUCED = pca.fit_transform(TRAIN)
    print('PCA : shape of digits reduced dataset: ', np.shape(REDUCED))

    ##### PERFORM AIC TO SEARCH BEST NUMBER OF COMPONENTS ################
    min_n_components = 1
    max_n_components = np.shape(REDUCED)[0]
    n_components = np.arange(min_n_components, max_n_components, 3)
    models = [GMM(n, covariance_type = 'full', random_state = 0) for n in n_components]
    aics = [model.fit(REDUCED).aic(REDUCED) for model in models]

    fig = plt.figure()
    plt.plot(n_components, aics);
    #plt.show()
    plt.savefig(path_datas + '/' + 'AIC_graph.png') # can hide DeprecationWarning

    mini = np.argmin(aics)
    best_nb_components = n_components[mini]
    print("Best number of components is : ", best_nb_components)

    ##### PERFORM GMM WITH BEST NUMBER COMPONENTS #######################
    gmm = GMM(best_nb_components, covariance_type = 'full', random_state = 0)
    gmm.fit(REDUCED)
    print('Convergence of GMM model fit to digits reduced dataset: ', gmm.converged_)

    # n_sample: sample of new surfaces
    n_sample = 30
    DATA_NEW = gmm.sample(n_sample, random_state = 0)
    print('Shape of random data : ', np.shape(DATA_NEW))

    # inverse transform of the PCA object to construct the new surfaces
    NEW = pca.inverse_transform(DATA_NEW)
    print('Shape of random data after inverse PCA : ', np.shape(NEW))

    thetas = np.linspace(0, 2*np.pi, n_radius)
    t_anevrisme = np.linspace(0, 1, 1000)

    for i in range(n_sample) :

        print("Saving sample {} and create aneurysm".format(i))

        SAMPLE = NEW[i,:]
        DILATATION_SAMPLE = SAMPLE[0]
        CENTERLINE_SAMPLE = SAMPLE[1:3*(degree+1)+1]
        RADIUS_SAMPLE = SAMPLE[3*(degree+1)+1:]

        ### CENTERLINE ##############
        step = int(len(CENTERLINE_SAMPLE)/3)
        coeffs_x = CENTERLINE_SAMPLE[0:step]
        coeffs_y = CENTERLINE_SAMPLE[step:2*step]
        coeffs_z = CENTERLINE_SAMPLE[2*step:]
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
        COORD *= DILATATION_SAMPLE
        print("------> Arc Length of the centerline : ", stats.arc_length(1,COORD))
        BSPLINE, TAN, NOR, BI = stats.frenet_frame(COORD)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(BSPLINE[:,0], BSPLINE[:,1], BSPLINE[:,2]) # Remember:  rows, cols = Z.shape
        bspline_plot_name = path_datas_random + '/random_bspline_' + str(i) + '.png'
        plt.savefig(bspline_plot_name)

        ### RADIUS ##################
        RADIUS_SAMPLE = RADIUS_SAMPLE.reshape(n_point, n_radius)
        #np.savetxt(path_datas + '/random_' + str(i) + ".csv", SAMPLE, delimiter = ',')

        fig = plt.figure()
        ax = fig.add_subplot(111,projection = '3d')
        ax.plot_surface(X, Y, RADIUS_SAMPLE.T, cmap = 'ocean') # Remember:  rows, cols = Z.shape
        surf_plot_name = path_datas_random + '/random_radius_' + str(i) + '.png'
        plt.savefig(surf_plot_name)

        liste_contour = []

        for k in range(len(COORD)):

            R_THETA = RADIUS_SAMPLE[k,:]
            TAB = np.hstack((thetas[np.newaxis].T, R_THETA[np.newaxis].T))

            #COORD
            C = BSPLINE[k,:]
            T = TAN[k,:]
            N = NOR[k,:]
            B = BI[k,:]

            #RECONSTRUCTION OF SHAPE OF THE SECTION
            PASSAGE = gmr.Matrice_de_passage(T, N, B)
            COORD_PLAN = ((np.dot(PASSAGE.T, C.T)).T)
            CONTOUR = gmr.Reconstruction_contour(COORD_PLAN, TAB, PASSAGE)
            liste_contour.append(CONTOUR)

        L = np.vstack(liste_contour)
        ANEVRISME = np.vstack((BSPLINE, L))
        gf.Write_csv(path_datas_random + '/' + "RANDOM_ANEURYSM_{}.csv".format(i), ANEVRISME, "x, y, z")

    end_time = time.time()
    print("Total time ", round(end_time - start_time, 2))
    print("")

    return 0
