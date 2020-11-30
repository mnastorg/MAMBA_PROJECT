import numpy as np
import meshio
import time
import trimesh
import os

from math import *
from scipy.interpolate import griddata
from scipy.spatial import distance

from tools import Files_Management as gf
from tools import Statistics_Utilities as stats

gf.Reload(gf)
gf.Reload(stats)

def Parametrization_analysis(path_parametrization):

    PARA = gf.Read_parametrization(path_parametrization)

    #WE ALWAYS BEGIN PARAMETRIZATION WITH HIGHEST Z COORDINATES
    PZ1 = PARA[0,2]
    PZ2 = PARA[-1,2]
    if PZ2 > PZ1 :
        PARA = np.flip(PARA, axis = 0)

    #COMPUTE THETAS
    print("-----> Number of thetas : ", np.shape(PARA[:,12:])[1])

    #COMPUTE NB POINT CENTERLINE
    print("-----> Number of centerline's points : ", np.shape(PARA)[0])

    #COMPUTE AVERAGE_RADIUS
    average_radius = np.mean(PARA[:,12:])
    print("-----> The average radius is {} mm. ".format(round(average_radius,2)))

    #COMPUTE ARC LENGTH
    arc_length = stats.arc_length(1.0, PARA[:,0:3])
    print("-----> The arc_length is {} mm. ".format(round(arc_length,2)))

    return PARA, average_radius, arc_length

def Extract_solution(path_solution):

    mesh = meshio.read(path_solution)

    POINTS = mesh.points

    pressure = mesh.point_data.get('pressure')
    velocity = mesh.point_data.get('velocity')

    print('---> Shape of file :  X / Y / Z / PREESURE / Vx /Vy / Vz')
    RESULT = np.hstack((POINTS, pressure[np.newaxis].T, velocity))

    return RESULT

def Mesh_cylinder(arc_length, radius, nb_centerline, nb_thetas, nb_subdivision):

    ### INITIALISATION THETAS AND SUBDIVISION
    theta = np.linspace(0, 2*np.pi, nb_thetas)
    subdivision = (np.linspace(0, 1, nb_subdivision))[1:]

    ### CREATION CENTERLINE CYLINDER
    z_coord = np.flip(np.linspace(0, arc_length, nb_centerline), axis = 0)
    CENTERLINE = np.zeros((nb_centerline,3))
    CENTERLINE[:,2] = z_coord

    ### FRENET FRAME ON CENTERLINE
    TAN = np.array([0,0,1])
    NOR = np.array([1,0,0])
    BI = np.array([0,1,0])

    ### CREATION MAILLAGE CYLINDRE
    mesh = []
    for i in range(np.shape(CENTERLINE)[0]) :
        A = CENTERLINE[i,:]
        mesh.append(A)
        T = (TAN)[np.newaxis].T
        N = (NOR)[np.newaxis].T
        B = (BI)[np.newaxis].T
        PASSAGE = np.hstack((N,B,T))

        A_PLAN = np.dot(PASSAGE.T, A)

        for angle in theta :
            rotation = np.asarray([np.cos(angle), np.sin(angle),0])
            point_sub = []
            #FROM INSIDE TO OUTSIDE
            for i in subdivision:
                point_sub.append(i*radius*rotation)
            new_points = A_PLAN + point_sub
            points_canon = np.dot(PASSAGE, new_points.T).T
            mesh.append(points_canon)

    MESH_CYLINDER = np.vstack(mesh)

    return MESH_CYLINDER

def Mesh_original(PARA, nb_thetas, nb_subdivision):

    theta = np.linspace(0, 2*np.pi, nb_thetas)
    subdivision = (np.linspace(0, 1, nb_subdivision))[1:]

    mesh = []

    for i in range(np.shape(PARA)[0]) :

        COORD = PARA[i,0:3]
        mesh.append(COORD)

        T = (PARA[i,3:6])[np.newaxis].T
        N = (PARA[i,6:9])[np.newaxis].T
        B = (PARA[i,9:12])[np.newaxis].T
        R_THETA = PARA[i,12:]

        PASSAGE = np.hstack((N,B,T))

        COORD_PLAN = np.dot(PASSAGE.T, COORD)

        for angle, dist in zip(theta, R_THETA):
            rotation = np.asarray([np.cos(angle), np.sin(angle), 0])
            point_sub = []
            for i in subdivision:
                point_sub.append(i*dist*rotation)
            new_points = COORD_PLAN + point_sub
            points_canon = np.dot(PASSAGE, new_points.T).T
            mesh.append(points_canon)

    return np.vstack(mesh)

def Interpolated_solution(MESH_ORIGINAL, SOLUTION):

    M = np.shape(SOLUTION)[1] - 3
    TAB = np.zeros((np.shape(MESH_ORIGINAL)[0], M))

    for i in range(M):
        TAB[:,i] = griddata(SOLUTION[:,0:3], SOLUTION[:,3+i], MESH_ORIGINAL, method = 'nearest')

    MAG_VELOCITY = np.linalg.norm(TAB[:,1:], axis = 1)[np.newaxis].T

    INTERPOL = np.hstack((TAB, MAG_VELOCITY))

    return INTERPOL

def Test_rotation_invariant(path_parametrization, path_solution):

    #### INITIALISATION
    PARA, average_radius, arc_length = Parametrization_analysis(path_parametrization)
    SOLUTION = Extract_solution(path_solution)

    #### COMPUTE CYLINDER
    CYLINDER = Mesh_cylinder(int(arc_length), int(average_radius), 1000, 160, 10)

    #### ORIGINAL MAPPING
    MESH_ORIGINAL = Mesh_original(PARA, 160, 10)
    INTERPOL = Interpolated_solution(MESH_ORIGINAL, SOLUTION)
    SOL_MESH_ORIGINAL = np.hstack((MESH_ORIGINAL, INTERPOL))
    SOL_CYLINDER = np.hstack((CYLINDER, INTERPOL))
    gf.Write_csv("SOL_ORIGINAL.csv", SOL_MESH_ORIGINAL, "x, y, z, pressure, Vx, Vy, Vz, magnitude")
    gf.Write_csv("SOL_CYLINDER.csv", SOL_CYLINDER, "x, y, z, pressure, Vx, Vy, Vz, magnitude")

    ### ROTATION
    thetax = 2*np.pi*np.random.random()
    print("X axis rotation of angle : ", thetax)
    thetay = 2*np.pi*np.random.random()
    print("Y axis rotation of angle : ", thetay)
    thetaz = 2*np.pi*np.random.random()
    print("Z axis rotation of angle : ", thetaz)
    ROTX = np.array([ [1, 0, 0], [0, np.cos(thetax), -np.sin(thetax)], [0, np.sin(thetax), np.cos(thetax)]])
    ROTY = np.array([ [np.cos(thetay), 0, np.sin(thetay)], [0, 1, 0], [-np.sin(thetay), 0, np.cos(thetay)]])
    ROTZ = np.array([ [np.cos(thetaz), -np.sin(thetaz), 0], [np.sin(thetaz), np.cos(thetaz), 0], [0, 0, 1]])

    ### ROTATION SOLUTION VTU
    POINTS = SOLUTION[:,0:3]
    VALUES = SOLUTION[:,3:]
    ROT_POINTS_X = np.dot(ROTX, POINTS.T)
    ROT_POINTS_Y = np.dot(ROTY, ROT_POINTS_X)
    ROT_POINTS_Z = np.dot(ROTZ, ROT_POINTS_Y)
    ROT_SOLUTION = np.hstack((ROT_POINTS_Z.T, VALUES))

    ### ROTATION PARAMETRIZATION
    P = PARA[:,0:3]
    T = PARA[:,3:6]
    N = PARA[:,6:9]
    B = PARA[:,9:12]
    RT = PARA[:,12:]
    ROT_P_X = np.dot(ROTX, P.T)
    ROT_T_X = np.dot(ROTX, T.T)
    ROT_N_X = np.dot(ROTX, N.T)
    ROT_B_X = np.dot(ROTX, B.T)
    ROT_P_Y = np.dot(ROTY, ROT_P_X)
    ROT_T_Y = np.dot(ROTY, ROT_T_X)
    ROT_N_Y = np.dot(ROTY, ROT_N_X)
    ROT_B_Y = np.dot(ROTY, ROT_B_X)
    ROT_P_Z = np.dot(ROTZ, ROT_P_Y)
    ROT_T_Z = np.dot(ROTZ, ROT_T_Y)
    ROT_N_Z = np.dot(ROTZ, ROT_N_Y)
    ROT_B_Z = np.dot(ROTZ, ROT_B_Y)

    ROT_PARA = np.hstack((ROT_P_Z.T, ROT_T_Z.T, ROT_N_Z.T, ROT_B_Z.T, RT))

    #### MAPPING GEO ROTATION
    MESH_ROT = Mesh_original(ROT_PARA, 160, 10)
    INTERPOL_ROT = Interpolated_solution(MESH_ROT, ROT_SOLUTION)
    SOL_MESH_ROT = np.hstack((MESH_ROT, INTERPOL_ROT))
    SOL_CYLINDER_ROT = np.hstack((CYLINDER, INTERPOL_ROT))
    gf.Write_csv("SOL_ORIGINAL_ROT.csv", SOL_MESH_ROT, "x, y, z, pressure, Vx, Vy, Vz, magnitude")
    gf.Write_csv("SOL_CYLINDER_ROT.csv", SOL_CYLINDER_ROT, "x, y, z, pressure, Vx, Vy, Vz, magnitude")

    #### COMPUTE ERROR
    error = np.linalg.norm(SOL_CYLINDER[:,-1] - SOL_CYLINDER_ROT[:,-1])
    print("L'erreur est de ", error)

    return 0

def steady_simulation(path):

    tree = sorted([f for f in os.listdir(path) if not f.startswith('.')], key = str.lower)

    results = tree[:-1]

    parametrization = tree[-1]

    PARA, average_radius, arc_length = Parametrization_analysis(path + '/' + parametrization)

    solution = []

    for i in results :
        print("Extraction solution ", i)
        SOLUTION = Extract_solution(path + '/' + i)
        solution.append(SOLUTION)

    MESH_ORIGINAL = Mesh_original(PARA, 160, 10)

    cylinder_sol = []
    for i in solution :
        INTERPOL = Interpolated_solution(MESH_ORIGINAL, i)
        cylinder_sol.append(INTERPOL[:,-1])

    return cylinder_sol
