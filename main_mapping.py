#############################################################################################################################
############################################## PACKAGES IMPORTATION #########################################################
#############################################################################################################################

#PACKAGE SYSTEM
import shutil
import os
import time

#PACKAGE MATHS
import numpy as np
from math import *

#TOOLS FILES
from tools import Gestion_Fichiers as gf
from tools import BSplines_Utilities as bs
from tools import Statistiques_Utilities as stats

#GEOMETRY_UTILITIES FILES
from geometry_utilities import Mapping_Utilities as map

#RELOAD ALL FILES (IF CHANGES IN ORIGINAL FILES)
gf.Reload(gf)
gf.Reload(bs)
gf.Reload(stats)
gf.Reload(map)

#################################################################################################################################
############################################## SHORT EXPLANATION ################################################################
#################################################################################################################################

#FUNCTION THAT BUILD THE MAPPING FROM SIMVSACULAR SOLUTION A CYLINDER USING PARAMETRIZATION FILE
#YOU NEED TO COPY SV RESULTS IN YOUR RESPECTIV FOLDER OF YOUR PATIENTS DATA

#################################################################################################################################
############################################## MAIN FUNCTION PROGRAMM ###########################################################
#################################################################################################################################

def Main_mapping(path_patients_datas):

    # PARAMETERS USED IN PARAMETRIZATION
    nb_centerline = 1000
    nb_thetas = 160
    nb_subdivision = 10

    tree = sorted([f for f in os.listdir(path_patients_datas) if not f.startswith('.')], key = str.lower)

    list_parametrization = []
    list_radius = []
    list_arc_length = []
    list_solution = []

    print('--------------------------------------------------------------------------------------------------')
    print('--------------------------------------- ANALYSING ALL FILES --------------------------------------')
    print('--------------------------------------------------------------------------------------------------')

    start_reading = time.time()
    for folder in tree :

        print('-------------------- READING PATIENT {} --------------------'.format(folder))
        path_para = path_patients_datas + '/' + folder + '/' + folder + '_parametrization.csv'
        path_sol = path_patients_datas + '/' + folder + '/' + folder + '_all_results_00100.vtu'

        print('--> Analysis of parametrization file')
        start = time.time()
        PARA, average_radius, arc_length = Mapping_Utilities.Parametrization_analysis(path_para)
        list_parametrization.append(PARA)
        list_radius.append(average_radius)
        list_arc_length.append(arc_length)
        end = time.time()
        print('--> Time reading parametrization : ', round(end - start , 2))

        print('--> Reading and saving solution (vtu) file')
        start = time.time()
        SOLUTION = Mapping_Utilities.Extract_solution(path_sol)
        list_solution.append(SOLUTION)
        end = time.time()
        print('--> Time reading solution (vtu) file : ', round(end - start , 2))

    end_reading = time.time()
    print('-> Total time to read all files : ', round(end_reading - start_reading, 2))

    print('--------------------------------------------------------------------------------------------------')
    print('--------------------------------- SETTING PARAMETERS FOR CYLINDER --------------------------------')
    print('--------------------------------------------------------------------------------------------------')

    #GLOBAL PARAMETERS OVERALL GEOMETRIES
    radius = np.mean(list_radius)
    print("-> The mean of radius on all geometries is {}, rounded at {}.".format(round(radius,2), int(radius)))
    arc_length = np.mean(list_arc_length)
    print("-> The mean of arc length on all geometries is {} rounded at {}".format(round(arc_length,2), int(arc_length)))

    #BUILD CYLINDER FOR ALL GEOMETRIES
    print('-> Building the cylinder (reference geometry) to map all solutions')
    start_cylinder = time.time()
    CYLINDER = Mapping_Utilities.Mesh_cylinder(arc_length, radius, nb_centerline, nb_thetas, nb_subdivision)
    end_cylinder = time.time()
    print('-> Total time to build cylinder : ', round(end_cylinder - start_cylinder, 2))

    print('--------------------------------------------------------------------------------------------------')
    print('------------------------------ BUILDING MAPPINGS FOR ALL GEOMETRIES ------------------------------')
    print('--------------------------------------------------------------------------------------------------')

    for i in range(len(list_parametrization)):
        start_map = time.time()
        print('-------------------- PATIENT {} --------------------'.format(tree[i]))

        path_original = path_patients_datas + '/' + tree[i] + '/' + tree[i] + '_interpolated_solution.csv'
        path_mapping = path_patients_datas + '/' + tree[i] + '/' + tree[i] + '_mapping.csv'

        #BUILDING MESH FROM PARAMETRIZATION
        print('--> Building the original mesh')
        MESH_ORIGINAL = Mapping_Utilities.Mesh_original(list_parametrization[i], nb_thetas, nb_subdivision)
        print('--> Interpolation of the solution')
        INTERPOL = Mapping_Utilities.Interpolated_solution(MESH_ORIGINAL, list_solution[i])
        print('--> Writing files in their folders')
        SOL_MESH_ORIGINAL = np.hstack((MESH_ORIGINAL, INTERPOL))
        SOL_CYLINDER = np.hstack((CYLINDER, INTERPOL))
        gf.Write_csv(path_original, SOL_MESH_ORIGINAL, "x, y, z, pressure, Vx, Vy, Vz, magnitude")
        gf.Write_csv(path_mapping, SOL_CYLINDER, "x, y, z, pressure, Vx, Vy, Vz, magnitude")
        end_map = time.time()
        print('-> Total time to map this patient : ', round(end_map - start_map, 2))

    print('-> Total time to map all geometries : ', round(end_map - end_cylinder, 2))

    print('-> Total time of mapping programm: ', round(end_map - start_reading, 2))

    return 0
