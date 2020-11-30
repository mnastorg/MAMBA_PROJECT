##################################################################################################
############### MODULE POUR GERER LECTURE ECRITURE DES FICHIERS CSV ##############################
##################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib

def Read_csv_xyz(file_csv_pcl):
    """Retourne un tableau qui lit le squelette de Nurea avec les paramètre x, y, z, label, isSail, isLeaf"""
    #LECTURE DU FICHIER CSV
    DF = pd.read_csv(file_csv_pcl)
    #LECTURE DES PARAMETRES QUI NOUS INTERESSENT
    x = np.asarray(DF['# x'])
    y = np.asarray(DF[' y'])
    z = np.asarray(DF[' z'])
    #CREATION DU TABLEAU RELATIF AU CSV
    if (len(x) == len(y) and len(x)==len(z)):
        PCL = np.zeros((len(x),3))
        PCL[:,0] = x
        PCL[:,1] = y
        PCL[:,2] = z
    else :
        print("ERREUR PAS LE MEME NBR DE COORDONNEES")

    return PCL

def Read_parametrization(file_parametrization):

    DF = pd.read_csv(file_parametrization, header = None)

    return np.asarray(DF)

def Read_csv_nurea(file_nurea):
    """Retourne un tableau qui lit le squelette de Nurea avec les paramètre x, y, z, label, isSail, isLeaf"""
    #LECTURE DU FICHIER CSV
    DF = pd.read_csv(file_nurea)
    #LECTURE DES PARAMETRES QUI NOUS INTERESSENT
    x = np.asarray(DF[' x'])
    y = np.asarray(DF[' y'])
    z = np.asarray(DF[' z'])
    lab = np.asarray(DF[' Label'])
    is_sail = np.asarray(DF[' isBifurcation'])
    is_leaf = np.asarray(DF[' isLeaf'])
    #CREATION DU TABLEAU RELATIF AU CSV
    if (len(x) == len(y) and len(x)==len(z)):
        NUREA = np.zeros((len(x),6))
        NUREA[:,0] = x
        NUREA[:,1] = y
        NUREA[:,2] = z
        NUREA[:,3] = lab
        NUREA[:,4] = is_sail
        NUREA[:,5] = is_leaf
    else :
        print("ERREUR PAS LE MEME NBR DE COORDONNEES")

    return NUREA

def Read_SimVascular_histor(folder_name):

    file = open('patients_datas/' + folder_name + '/' + folder_name + '_histor.dat')

    lst = []
    for line in file :
        lst += [line.split()]

    iterations = [float(x[0]) for x in lst]
    time = [float(x[1]) for x in lst]
    residuals = [float(x[2]) for x in lst]
    residual_entropy_velocity = [float(x[5]) for x in lst]
    residual_entropy_pressure= [float(x[6]) for x in lst]

    i = 0
    index1 = []
    index2 = []

    while i < len(iterations)-1 :

        if iterations[i] == iterations[i+1] :
            index1.append(i)
            index2.append(i+1)
            i += 2
        else :
            index1.append(i)
            i+= 1

    iterations1 = [iterations[i] for i in index1]
    residuals1 = [residuals[i] for i in index1]
    residual_entropy_velocity1 = [residual_entropy_velocity[i] for i in index1]
    residual_entropy_pressure1 = [residual_entropy_pressure[i] for i in index1]

    iterations2 = [iterations[i] for i in index2]
    residuals2 = [residuals[i] for i in index2]
    residual_entropy_velocity2 = [residual_entropy_velocity[i] for i in index2]
    residual_entropy_pressure2 = [residual_entropy_pressure[i] for i in index2]

    plt.figure()
    plt.plot(iterations1, residuals1, label = 'Residual 1')
    plt.plot(iterations2, residuals2, label = 'Residual 2')
    plt.yscale('log')
    plt.legend()
    plt.title('Non-linear residuals / Iterations')
    plt.savefig('patients_datas/' + folder_name + '/' + folder_name + '_residual_error.jpg')
    plt.show()

    #plt.figure()
    #plt.plot(iterations1[5:], residual_entropy_velocity1[5:], label = 'Residual velocity it 1')
    #plt.plot(iterations2[5:], residual_entropy_velocity2[5:], label = 'Residual pressure it 2')
    #plt.yscale('log')
    #plt.legend()
    #plt.title('Velocity entropy norm residuals / Iterations 1-2')
    #plt.figure()
    #plt.plot(iterations1[5:], residual_entropy_pressure1[5:], label = 'Residual pressure it 1')
    #plt.plot(iterations2[5:], residual_entropy_pressure2[5:], label = 'Residual pressure it 2')
    #plt.yscale('log')
    #plt.legend()
    #plt.title('Pressure entropy norm residuals / Iterations 1-2')

    plt.show()

    return 0

def Write_csv(NOM, CONCAT, HEAD):
    """Ecrit un fichier csv 'NOM' issu de CONCAT avec un header"""
    np.savetxt(NOM, CONCAT, delimiter = ", ", header = HEAD)

def Reload(module):
    importlib.reload(module)
