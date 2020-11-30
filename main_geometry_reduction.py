#############################################################################################################################
############################################## PACKAGES IMPORTATION #########################################################
#############################################################################################################################

#PACKAGE SYSTEM
import os
import meshio
import shutil

#PACKAGE MATHS AND PLOT
import numpy as np
from math import *
import matplotlib.pyplot as plt

#TOOLS FILES
from tools import Files_Management as gf
from tools import BSplines_Utilities as bs
from tools import Statistics_Utilities as stats

#MODEL_REDUCTION_UTILITES FILES
from model_reduction_utilities import Geometric_Model_Reduction as gmr

#GEOMETRY_UTILITIES FILES
from geometry_utilities import Geometry_Treatment as geo_treat

#RELOAD ALL FILES (IF CHANGES IN ORIGINAL FILES)
gf.Reload(gf)
gf.Reload(bs)
gf.Reload(stats)
gf.Reload(gmr)
gf.Reload(geo_treat)

#################################################################################################################################
############################################## SHORT EXPLANATION ################################################################
#################################################################################################################################

#THIS FUNCTION BUILD A GEOMETRIC MODEL REDUCTION OF A SET OF INITIAL AORTIC ANEURYSM GEOMETRIES.
#IT TAKES AS INPUT THE FOLDER IN WHICH YOU STORED DATA TO ANALYSE (DATA USUALLY CREATED FROM main_pre_processing).
#nb_generation IS THE NUMBER OF RANDOM ANEURYSM TO GENERATE.
#IF NEEDED mesh_extractor CONVERTS THE POINT CLOUD TO A 3D SURFACE MESH. TO BE USED IN CFD SIMULATOR, UNCOMMMENT
#LAST LINE (NOT RECOMMANDED AT FIRST BECAUSE IT TAKES A VERY LONG TIME TO REMESH).
#LINE 112, read_file IS SET False. TAKE A LOOK AT THE FUNCTION TO UNDERSTAND WHY IT IS RECOMMANDED TO TURN IT True AFTER
#FIRST RUN OF THE ALGORITHM.


#################################################################################################################################
############################################## MAIN FUNCTION PROGRAMM ###########################################################
#################################################################################################################################

def Main_geo_reduction(path_patients_data, nb_generation = 20, mesh_extractor = False):

    print("---------------------------------- START PROGRAMM ----------------------------------")

    # EXTRACT DATA FROM FOLDER "path_patients_data". READ ALL PARAMETRIZATION FILES.

    print("--> DATAS EXTRACTION FROM PARAMETRIZATION FILES")

    folder = sorted([f for f in os.listdir(path_patients_data) if not f.startswith('.')], key = str.lower)
    print("----> Number of patients considered : {}".format(len(folder)))

    liste_control = []
    liste_r_theta = []
    liste_originale_centerline = []

    thetas = np.linspace(0, 2*np.pi, 160)

    #WE LOOP ON EACH PARAMETRIZATION FILE AND EXTRACT RELEVANT INFORMATION
    for i in folder :

        #CHANGE path IF YOU GAVE ANOTHER NAME TO PARAMETRIZATION FILE
        path = path_patients_data + '/' + i + '/' + i + '_parametrization.csv'
        PARA = gf.Read_parametrization(path)

        #EXTRACTION CENTERLINE
        CENTERLINE = PARA[:,0:3]
        liste_originale_centerline.append(CENTERLINE)

        #EXTRACTION CONTROL POINTS TO USE PROCRUSTES ANALYSES AND RECONSTRUCTION B-SPLINES
        extract = np.linspace(0, np.shape(CENTERLINE)[0]-1, 10, dtype = 'int')
        CONTROL = CENTERLINE[extract,:]
        liste_control.append(CONTROL)

        #EXTRACTION OF FUNCTION R_THETA : DISTANCE TO ANGLE
        r_theta = PARA[:,12:]
        liste_r_theta.append(r_theta)

    #UNCOMMENT NEXT LINE IF YOU WANT TO STORE ORIGINAL CENTERLINE IN A FILE
    #gf.Write_csv("ORIGINAL_CENTERLINES.csv", np.vstack(liste_originale_centerline), "x, y, z")

    #############################################################################
    ################ PARAMETERS FOR GEOMETRIC MODEL REDUCTION ###################
    #############################################################################
    #EPSILON TO AUTOMATICALLY EXTRACT NUMBER OF POD MODES
    epsilon_c = 1.e-2
    epsilon_w = 1.e-2
    epsilon_r = 1.e-2
    #PARAMETERS TO RECONSTRUCT ANEURYSMS
    nb_sections = 500
    nb_thetas = 160
    #############################################################################
    #############################################################################
    #############################################################################

    print("")
    print("--> EXTRACTION CENTERLINE REDUCED BASIS")

    dilatation, PHI_CENTERLINE, COEFFS_CENTERLINE = gmr.Centerline_pod(liste_control, epsilon = epsilon_c)

    print("")
    print("--> EXTRACTION WALL AND RADIUS REDUCED BASIS")

    PHI_WALL, PHI_RAYON, COEFFS_WALL, COEFFS_RAYON = gmr.Wall_pod(thetas, liste_r_theta, epsilon_wall = epsilon_w, epsilon_radius = epsilon_r, read_file = False)

    print("")
    print("--> GENERATION OF RANDOM AORTIC ANEURYSMS GEOMETRIES")

    for i in range(nb_generation) :

        print("----> ANEURYSM NUMBER : ", i)

        #GIVE A NAME FOR THE ANEURYSM TO SAVE IT
        name = "ANE_" + str(i)
        #GIVE NAME OF FOLDER IN WHICH TO STORE ANEURYSMS
        path_to_save = "results/aneurysms_generation/"

        #COMPUTE RANDOM ANEURYSM
        ANEVRISME = gmr.Aneurysm_generator(PHI_CENTERLINE, PHI_WALL, PHI_RAYON, COEFFS_CENTERLINE, COEFFS_WALL, COEFFS_RAYON, dilatation, nb_sections, nb_thetas)
        gf.Write_csv(path_to_save + "reconstruction_" + name + ".csv", ANEVRISME, "x, y, z")

        if mesh_extractor == True :
            CONTOUR = ANEVRISME[nb_coupures:,:]
            geo_treat.Mesh_generation(CONTOUR, path_to_save + "mesh_contour_" + name + ".stl", nb_sections, nb_thetas)
            geo_treat.Read_and_Smooth(path_to_save + "mesh_contour_" + name + ".stl", path_to_save + "mesh_smooth_" + name + ".stl", coeff_smooth = 0.001)
            #UNCOMMENT NEXT LINE IF YOU WANT TO REMESH FILES (WARNING : CAN BE VERY LOOONG TO COMPUTE!)
            #geo_treat.Surface_Remesh(path_to_save + "mesh_smooth_" + name + ".stl", path_to_save + "mesh_remesh_" + name + ".stl", target_edge_length = 0.3, nb_iterations = 10)

    return 0
