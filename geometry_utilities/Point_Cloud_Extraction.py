import numpy as np
import trimesh
import time
import skfmm
import scipy.signal
from sklearn.neighbors import LocalOutlierFactor

from tools import Files_Management as gf

####################################################################################
############################### MAIN POINT CLOUD EXTRACTION ########################
####################################################################################

def Extraction(file_stl, pitch, seuil):

    start_time = time.time()

    aorte = trimesh.load_mesh(file_stl)

    #TRAITEMENT STL INITIAL
    VOX,vox = Voxelisation(aorte, pitch)

    print(" ---> Number of surface voxels : ", vox.filled_count)

    FILL, fill = Voxel_fill(vox)

    indices = Indices_matrix(fill)
    indices_geo = Indices_geo(fill, indices)

    HOL, hol = Voxel_surface(fill)

    print(" ---> Total number of voxels after fill : ", fill.filled_count)
    print(" ---> Grid size : ", fill.shape)

    #MAT_FM = Matrix_fast_marching_1(VOX, FILL)
    MAT_FM = Matrix_fast_marching_2(VOX, FILL, HOL)

    #EXTRACTION DES INDICES RELATIFS AUX POINTS

    step1 = time.time()
    print(" ---> Voxelization time : ", round(step1 - start_time, 2))

    #LEVELSET SANS LES POINTS EXTERIEURS A LA GEOMETRIE
    LEVELSET = Level_set(MAT_FM)
    INDICES_LEVELSET = Ajout_indices(LEVELSET, indices_geo, FILL)

    step2 = time.time()
    print(" ---> Fast Marching time : ", round(step2 - step1, 2))

    #NOYAU POUR LA REGULARISATION
    G3 = Noyau_gaussien_3D(0.5,2)
    REG = Convolution(LEVELSET, G3)
    INDICES_REG = Ajout_indices(REG, indices_geo, FILL)

    step3 = time.time()
    print(" ---> Gaussian convolution time : ", round(step3 - step2, 2))

    #NOYAU POUR FILTRE LAPLACIEN
    L7 = Noyau_laplacien_7()
    LAP = Convolution(REG, L7)
    INDICES_LAP = Ajout_indices(LAP, indices_geo, FILL)

    step4 = time.time()
    print(" ---> Laplacian convolution time : ", round(step4 - step3, 2))

    #EXTRACTION DU NUAGE DE POINT PAR UN SEUIL PRECIS
    PCL = Extraction_pcl(INDICES_LAP, seuil)

    step5 = time.time()
    print(" ---> PCL extraction time : ", round(step5 - step4, 2))

    clf = LocalOutlierFactor(n_neighbors = 30)
    pred = clf.fit_predict(PCL)
    TO_REMOVE = np.hstack((PCL,pred[np.newaxis].T))
    FINAL = TO_REMOVE[TO_REMOVE[:,3]>0]

    step6 = time.time()
    print(" ---> Outliers removals time : ", round(step5 - step6, 2))
    print(" ---> Total time to extract PCL : ", round(step6 - start_time, 2))

    return FINAL[:,0:3], INDICES_LEVELSET

####################################################################################
########################## UTILITIES FOR POINT CLOUD EXTRACTION  ###################
####################################################################################

def Voxelisation(mesh, pitch):
    """Retourne VOX format np.array et vox format trimesh correspondant à la
    voxelisation de geom avec une précision pitch"""
    vox = mesh.voxelized(pitch)
    VOX = vox.matrix

    return VOX, vox

def Voxel_fill(vox):
    """Retourne FILL format np.array et fill format trimesh. Les matrices remplisse
    vox avec des voxel (on cherche à remplir la triangulation de surface initiale)"""
    fill = vox.fill(method = 'orthographic')
    FILL = fill.matrix

    return FILL, fill

def Voxel_surface(fill):
    hol = fill.hollow()
    HOL = hol.matrix

    return HOL, hol

def Matrix_fast_marching_1(VOX, FILL):
    """Retourne la matrice GEOM_DIST à utiliser dans Distance.py pour appliquer
    la fonction de FastMarching python. 0 si frontière, 1 intérieur et -1 extérieur"""
    GEOM_DIST = 0*FILL
    GEOM_DIST[FILL == False] = -1
    GEOM_DIST[FILL != VOX] = 1

    return GEOM_DIST

def Matrix_fast_marching_2(VOX, FILL, HOL):
    GEOM_DIST = 0*FILL
    GEOM_DIST[HOL == False] = 1
    GEOM_DIST[FILL == False] = -1
    return GEOM_DIST

def Indices_matrix(fill):
    """Retourne un tableau des coordonnées correspondant aux voxels TRUE (attention
    il faudra s'adapter pour que nos tableaux soient de la taille d'indices_geo)"""
    indices = fill.sparse_indices
    return  indices

def Indices_geo(fill,indices):
    """Retourne à partir des indices matriciels les indices de la geo originale"""
    indices_geo = fill.indices_to_points(indices)
    return indices_geo

def Level_set(BORDURE):
    """ Retourne la matrice DISTZERO de fonction distance relative à GEOM_DIST
    où les valeurs aux points extérieurs sont 0"""
    DIST = skfmm.distance(BORDURE, order = 1)
    DISTZERO = DIST.copy()
    DISTZERO[DISTZERO < 0] = 0

    return DISTZERO

def Ajout_indices(LEVELSET, indices_geo, FILL):
    """Retourne un tableau près à l'affichage de type x/y/z/scalaire avec scalaire
    issu de LEVELSET et x/y/z issus de indices_geo. FILL permet de supprimer les éléments
    extérieurs à la geo dans LEVELSET"""

    [n1, n2, n3] = np.shape(LEVELSET)

    LEVELSET = LEVELSET.reshape(n1*n2*n3)
    FILL = FILL.reshape(n1*n2*n3)
    todelete = np.where(FILL == False)
    toconcatenate = np.delete(LEVELSET, todelete)
    toconcatenate = toconcatenate.reshape((np.shape(toconcatenate)[0],1))
    CONCAT = np.concatenate((indices_geo, toconcatenate), axis = -1)

    return CONCAT

def Noyau_gaussien_1D(sigma, nb):
    """Retourne le noyau gaussien 1D"""
    x = np.linspace(-3*sigma, 3*sigma, 2*(nb)+1)
    x1= np.exp(-x**2/(2*sigma**2))
    x1 = x1/np.sum(x1)

    return x1

def Noyau_gaussien_2D(sigma, nb):
    """Retourne le noyau gaussien_2D"""
    x1 = Noyau_gaussien_1D(sigma, nb)

    Fx1 = np.fft.fft(x1)
    Fx1 = Fx1[np.newaxis]
    Fx2 = np.transpose(Fx1)

    Fx1x2 = np.dot(Fx2,Fx1)

    x1x2 = np.real(np.fft.ifft2(Fx1x2))

    return x1x2

def Noyau_gaussien_3D(sigma, nb):
    """Retourne le noyau gaussien_3D"""
    x = np.linspace(-3*sigma, 3*sigma, 2*(nb)+1)
    x1 = Noyau_gaussien_1D(sigma, nb)
    x2x3 = Noyau_gaussien_2D(sigma, nb)

    Fx1 = np.fft.fft(x1)
    Fx1 = Fx1[np.newaxis]

    Fx2x3 = np.fft.fft2(x2x3)
    Fx2x3 = np.transpose(Fx2x3[np.newaxis])

    Fx1x2x3 = np.dot(Fx2x3,Fx1)
    x1x2x3 = np.real(np.fft.ifftn(Fx1x2x3))

    return x1x2x3

def Noyau_laplacien_7():
    """Retourne le noyau du Laplacien discret pour 7 points"""
    return np.array([[[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]],
                    [[0, 1, 0],
                    [1, -6, 1],
                    [0, 1, 0]],
                    [[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]]])

def Noyau_laplacien_27():
    """Retourne le noyau du Laplacien discret pour 27 points"""
    return np.array([[[2/26, 3/26, 2/26],
                         [3/26, 6/26, 3/26],
                         [2/26, 3/26, 2/26]],
                        [[3/26, 6/26, 3/26],
                         [6/26, -88/26, 6/26],
                         [3/26, 6/26, 2/26]],
                        [[2/26, 3/26, 2/26],
                         [3/26, 6/26, 2/26],
                         [2/26, 3/26, 2/26]]])

def Convolution(MAT, kernel):
    """Fait la convolution entre la matrice MAT et le noyau kernel"""
    return scipy.signal.oaconvolve(MAT, kernel, mode = "same")

def Extraction_pcl(CONCAT_LAP, seuil):
    """ Retourne le squelette à partir du laplacien de la fonction distance. On se
    base sur l'extraction des points inférieur à p dans [0,1] fois le min """
    min = np.min(CONCAT_LAP[:,3])
    PCL = CONCAT_LAP[CONCAT_LAP[:,3] <= seuil*min]

    return PCL[:,0:3]
