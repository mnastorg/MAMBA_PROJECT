#############################################################################################################################
############################################## PACKAGES IMPORTATION #########################################################
#############################################################################################################################

#SYSTEM PACKAGES
import shutil
import os
import time

#MATHS PACKAGES
import numpy as np
from math import *

#TOOLS PACKAGES
from tools import Files_Management as gf
from tools import BSplines_Utilities as bs
from tools import Statistics_Utilities as stats

#GEOMETRY UTILITIES FILES
from geometry_utilities import Geometry_Treatment as geo_treat
from geometry_utilities import Parametrization_Utilities as para

gf.Reload(gf)
gf.Reload(stats)
gf.Reload(bs)

gf.Reload(geo_treat)
gf.Reload(para)

#################################################################################################################################
############################################## SHORT EXPLANATION ################################################################
#################################################################################################################################

#THIS FUNCTION BUILDS A FOLDER (GIVEN NAME IN path_datas LINE 49) WHICH CONTAINS RELEVANT DATA EXTRACTED FROM 3D IMAGES
#NIFTI FORMAT. IT TAKES AS INPUT THE PATH TO THE FOLDER WHICH CONTAINS NIFTI FILES. IF YOU WANT TO ADD EXTENSIONS TO THE
#GEOMETRY SET add_extension TO True.
#IT PERFORMS THE SEVERAL STEPS :
# - CONVERT NIFTI FILE TO 3D SURFACE MESH FORMAT STL
# - SMOOTH THE OBTAINED MESH
# - EXTRACT CENTERLINE (MANUALLY : PLACE MOUTH ON ONE EXTREMITY OF THE GEO AND PRESS SPACE AND THEN Q. REPEAT THE OPERATION FOR
#SECOND EXTREMITY OF THE GEOMETRYE)
# - PERFORM PARAMETRIZATION ACCORDING TO METHOD DEVELOP IN THE PDF
# - OPEN THE GEOMETRY AT THE EXTREMITY AND REMESH IT
# - ADD EXTENSION AT THE EXTREMITY IF NEEDED

#YOU CAN USE RESULTS IN CFD SIMULATION SOFTWARE WITH THE OPEN_REMESH FILE (SimVascular for instance)
#YOU CAN USE PARAMETRIZATION FILES TO RUN GEOMETRIC MODEL REDUCTION.

#################################################################################################################################
############################################## MAIN FUNCTION PROGRAMM ###########################################################
#################################################################################################################################

def Main_pre_processing(path_nifti_all, add_extension = False) :

    print("---------------------------------- START FILE TREATMENT ----------------------------------")

    #READ THE FOLDER
    folder = sorted([f for f in os.listdir(path_nifti_all) if not f.startswith('.')], key = str.lower)
    print("Number of nifti files in folder : {} \n".format(len(folder)))

    #WRITE HERE FOLDER NAME WHERE YOU WANT TO STORE YOUR DATA ! (MINE IS CALLED patients_data)
    path_datas = "patients_data_large/"

    #READ EACH FILE IN FOLDER
    for file in folder :

        print("------------------------------ Treatment of patient {} ------------------------------ ".format(file))

        start_time = time.time()

        name_file = file[:4]

        print(" -> Creation of folder named {} and initialisation of all paths".format(name_file))

        #CREATION FOLDER
        dir = path_datas + name_file
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

        #ALL PATH TO BE USED
        path_nifti_file = path_nifti_all + '/' + file
        path_marching_cube = dir + '/' + name_file + '_marching_cube.stl'
        path_mesh_closed = dir + '/' + name_file + '_mesh_closed.stl'
        path_centerline_vtp = dir + '/' + name_file + '_centerline.vtp'
        path_centerline_csv = dir + "/" + name_file + "_centerline_bspline.csv"
        path_control_csv = dir + "/" + name_file + "_centerline_control.csv"
        path_parametrization = dir + '/' + name_file + '_parametrization.csv'
        path_reconstruction = dir + '/' + name_file + '_reconstruction.csv'
        path_mesh_opened = dir + '/' + name_file + '_mesh_opened.stl'
        path_mesh_opened_remesh = dir + '/' + name_file + '_mesh_opened_remesh.stl'
        path_extension = dir + '/' + name_file + '_mesh_extension.stl'


        ########################################################################
        ######################## PARAMETERS USED ###############################
        ########################################################################

        #SMOOTHING
        coefficient_smoothing = 0.001
        iterations_smoothing = 50

        #CENTERLINE EXTRACTION
        coefficient_centerline = 0.1
        iterations_centerline = 50

        #CONVERT CENTERLINE TO BSPLINES
        nb_control_points = 10
        nb_centerline_points = 200
        bspline_degree = 3

        #WALL PARAMETRIZATION
        degree_centerline = 3
        nb_centerline_points_parametrization = 3000
        nb_thetas = 200
        fourier_order = 5

        #REMESH THE OPENED GEOMETRY
        edge_length = 0.5
        iterations_remesh = 5

        ########################################################################
        ########################################################################
        ########################################################################

        print("")
        print(" -> Read and Convert Nifti file {} to STL format thanks to Marching-Cube algorithm".format(file))
        geo_treat.Mesh_from_Nifti(path_nifti_file, path_marching_cube, only_lumen = True)
        print("")

        print(" -> Read Marching-Cube STL file, Smooth it and Convert the result to STL format")
        geo_treat.Read_and_Smooth(path_marching_cube, path_mesh_closed, coeff_smooth = coefficient_smoothing, nb_iterations = iterations_smoothing)
        print("")

        print(" -> Extraction of Centerline, Smooth it and Convert the result to VTP format")
        geo_treat.Centerline_Extraction(path_mesh_closed, path_centerline_vtp, coeff_smooth = coefficient_centerline, nb_iterations = iterations_centerline)
        print("")

        print(" -> Conversion centerline .VTP to numpy format and save as .CSV file")
        CONTROL, CENTERLINE = geo_treat.Centerline_BSpline(path_centerline_vtp, nb_control = nb_control_points, nb_points = nb_centerline_points, degree = bspline_degree)
        gf.Write_csv(path_control_csv, CONTROL, "x, y, z")
        gf.Write_csv(path_centerline_csv, CENTERLINE, "x, y, z")
        print("")

        print(" -> Parametrization of the mesh named " + name_file +"_mesh_closed.stl")
        PARAMETRIZATION, RECONSTRUCTION = para.Parametrization(CENTERLINE, path_mesh_closed, degree = degree_centerline, nb_centerline = nb_centerline_points_parametrization, nb_thetas = nb_thetas, nb_modes_fourier = fourier_order)
        np.savetxt(path_parametrization, PARAMETRIZATION, delimiter = ", ")
        gf.Write_csv(path_reconstruction, RECONSTRUCTION, "x, y, z")
        print("")

        print(" -> Cut the geometry to the extremity to open it")
        geo_treat.Mesh_Slice(path_mesh_closed, PARAMETRIZATION, path_mesh_opened)
        print("")

        #print(" -> Remesh of the open mesh geometry")
        #geo_treat.Surface_Remesh(path_mesh_opened, path_mesh_opened_remesh, target_edge_length = edge_length, nb_iterations = iterations_remesh)

        if add_extension == True :
            print(" -> Add extension at the boudaries of the open mesh")
            geo_treat.Add_extension(path_mesh_opened_remesh, path_extension, extension_ratio = 10, target_edge_length = 0.5, nb_iterations = 5)

        end_time = time.time()
        print("Total time for current patient : ", round(end_time - start_time, 2))
        print("")
        print("\a")

    return 0
