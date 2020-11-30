import os
import sys

import numpy as np
import trimesh
import meshio

from vmtk import pypes
from vmtk import vmtkscripts

import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk

from scipy.spatial import distance

from tools import Files_Management as gf
from tools import BSplines_Utilities as bs

gf.Reload(gf)
gf.Reload(bs)

############################################################################################
################################### NIFTI READER ###########################################
############################################################################################

def Mesh_from_Nifti(nifti_file_to_read, path_writer, only_lumen = True):

    if not os.path.exists(nifti_file_to_read):
        print(f"File: {nifti_file_to_read} does not exist")
        sys.exit(1)

    #NIFIT READING AND CONVERSION TO NUMPY ARRAY
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(nifti_file_to_read)
    reader.Update()
    seg_image = reader.GetOutput()
    seg_image_array = vtkToNumpy(seg_image)

    #TREAT SEGMENTATION ACCORDING TO KEEP ONLY LUMEN
    #OR FULL ANEURYSM
    if only_lumen == True :
        # Remove all data except lumen
        seg_image_array[seg_image_array > 1] = 0

    else:
        # Convert thrombus data to lumen data.
        seg_image_array[seg_image_array > 1] = 1

    #NUMPY ARRAY TREATED AND CONVERSION TO VTK FILE
    seg_image_cleaned = numpyToVTK(seg_image_array)
    seg_image_cleaned.SetOrigin(seg_image.GetOrigin())
    seg_image_cleaned.SetSpacing(seg_image.GetSpacing())
    #seg_image_cleaned.SetDirectionMatrix(seg_image.GetDirectionMatrix())

    #MARCHING CUBES ON VOXELS TO CREATE THE MESH
    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputData(seg_image_cleaned)
    dmc.GenerateValues(1, 1, 1)
    dmc.Update()

    #WRITE THE MESH TO STL FORMAT
    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(dmc.GetOutputPort())
    writer.SetFileTypeToBinary()
    writer.SetFileName(path_writer)
    writer.Write()

def vtkToNumpy(data):
    #PASSAGE VTK TO NUMPY
    temp = vtk_to_numpy(data.GetPointData().GetScalars())
    dims = data.GetDimensions()
    numpy_data = temp.reshape(dims[2], dims[1], dims[0])
    numpy_data = numpy_data.transpose(2,1,0)

    return numpy_data

def numpyToVTK(data):
    #PASSAGE NUMPY TO VTK
    flat_data_array = data.transpose(2,1,0).flatten()
    vtk_data_array = numpy_to_vtk(flat_data_array)
    vtk_data = numpy_to_vtk(num_array=vtk_data_array, deep=True, array_type=vtk.VTK_FLOAT)
    img = vtk.vtkImageData()
    img.GetPointData().SetScalars(vtk_data)
    img.SetDimensions(data.shape)

    return img

############################################################################################
############################## FUNCTIONS TO USE VMTK PACKAGE ###############################
############################################################################################

def Read_and_Smooth(path_reader, path_writer, coeff_smooth = 0.001, nb_iterations = 50):

    #READ THE SURFACE
    print(" ---> Reading Marching-Cube surface")
    myReader = vmtkscripts.vmtkSurfaceReader()
    myReader.InputFileName = path_reader
    myReader.Format = 'stl'
    myReader.Execute()

    #SMOOTH THE SURFACE
    print(" ---> Smoothing surface with coeff {} and {} ierations".format(coeff_smooth, nb_iterations))
    mySmoother = vmtkscripts.vmtkSurfaceSmoothing()
    mySmoother.Surface = myReader.Surface
    mySmoother.PassBand = coeff_smooth
    mySmoother.NumberOfIterations = nb_iterations
    mySmoother.Execute()

    #WRITE TO STL FILE
    print(" ---> Writing results to STL file ")
    myWriter = vmtkscripts.vmtkSurfaceWriter()
    myWriter.Surface = mySmoother.Surface
    myWriter.OutputFileName = path_writer
    myWriter.Format = 'stl'
    myWriter.Execute()

def Centerline_Extraction(path_reader, path_writer, coeff_smooth = 0.001, nb_iterations = 50):

    #READ THE SURFACE
    print(" ---> Reading Mesh Closed STL surface")
    myReader = vmtkscripts.vmtkSurfaceReader()
    myReader.InputFileName = path_reader
    myReader.Format = 'stl'
    myReader.Execute()

    print("---> Computing Centerline")
    myCenterline = vmtkscripts.vmtkCenterlines()
    myCenterline.Surface = myReader.Surface
    myCenterline.AppendEndPoints = 1
    myCenterline.Resampling = 1
    myCenterline.ResamplingStepLength = 0.2
    myCenterline.Execute()

    myCenterlineSmoother = vmtkscripts.vmtkCenterlineSmoothing()
    myCenterlineSmoother.Centerlines = myCenterline.Centerlines
    myCenterlineSmoother.SmoothingFactor = coeff_smooth
    myCenterlineSmoother.NumberOfSmoothingIterations = nb_iterations
    myCenterlineSmoother.Execute()

    #WRITE TO STL FILE
    print(" ---> Writing results to VTP file ")
    myWriter = vmtkscripts.vmtkSurfaceWriter()
    myWriter.Surface = myCenterlineSmoother.Centerlines
    myWriter.OutputFileName = path_writer
    #myWriter.Format = 'vtp'
    myWriter.Execute()

def VTPCenterline_To_Numpy(path_reader):

    myReader = vmtkscripts.vmtkSurfaceReader()
    myReader.InputFileName = path_reader
    myReader.Execute()

    NumpyAdaptor = vmtkscripts.vmtkCenterlinesToNumpy()
    NumpyAdaptor.Centerlines = myReader.Surface
    NumpyAdaptor.Execute()

    numpyCenterlines = NumpyAdaptor.ArrayDict
    CENTERLINE = numpyCenterlines["Points"]

    return CENTERLINE

def Add_extension(path_reader, path_writer, extension_ratio = 10, target_edge_length = 0.7, nb_iterations = 5):

    myReader = vmtkscripts.vmtkSurfaceReader()
    myReader.InputFileName = path_reader
    myReader.Format = 'stl'
    myReader.Execute()

    myCenterline = vmtkscripts.vmtkCenterlines()
    myCenterline.Surface = myReader.Surface
    myCenterline.SeedSelectorName = 'openprofiles'
    myCenterline.Execute()

    myExtension = vmtkscripts.vmtkFlowExtensions()
    myExtension.Surface = myReader.Surface
    myExtension.Centerlines = myCenterline.Centerlines
    myExtension.AdaptiveExtensionLength = 1
    myExtension.AdaptiveExtensionRadius = 1
    myExtension.ExtensionMode = "boundarynormal"
    myExtension.ExtensionRatio = extension_ratio
    myExtension.Interactive = 0
    myExtension.Execute()

    mySmoother = vmtkscripts.vmtkSurfaceSmoothing()
    mySmoother.Surface = myExtension.Surface
    mySmoother.PassBand = 0.3
    mySmoother.NumberOfIterations = 5
    mySmoother.Execute()

    myRemesh = vmtkscripts.vmtkSurfaceRemeshing()
    myRemesh.Surface = mySmoother.Surface
    myRemesh.ElementSizeMode = 'edgelength'
    myRemesh.TargetEdgeLength = target_edge_length
    myRemesh.NumberOfIterations = nb_iterations
    myRemesh.Execute()

    myWriter = vmtkscripts.vmtkSurfaceWriter()
    myWriter.Surface = myRemesh.Surface
    myWriter.OutputFileName = path_writer
    myWriter.Format = 'stl'
    myWriter.Execute()

def Surface_Remesh(path_reader, path_writer, target_edge_length = 0.7, nb_iterations = 10):

    myReader = vmtkscripts.vmtkSurfaceReader()
    myReader.InputFileName = path_reader
    myReader.Format = 'stl'
    myReader.Execute()

    myRemesh = vmtkscripts.vmtkSurfaceRemeshing()
    myRemesh.Surface = myReader.Surface
    myRemesh.ElementSizeMode = 'edgelength'
    myRemesh.TargetEdgeLength = target_edge_length
    myRemesh.NumberOfIterations = nb_iterations
    myRemesh.Execute()

    myWriter = vmtkscripts.vmtkSurfaceWriter()
    myWriter.Surface = myRemesh.Surface
    myWriter.OutputFileName = path_writer
    myWriter.Format = 'stl'
    myWriter.Execute()

############################################################################################
################################### CENTERLINE IMPROVEMENT #################################
############################################################################################

def Centerline_BSpline(path_reader, nb_control = 10, nb_points = 200,  degree = 5):

    CENTERLINE = VTPCenterline_To_Numpy(path_reader)
    NEW = CENTERLINE[20:np.shape(CENTERLINE)[0]-20 , :]

    N = np.shape(NEW)[0]
    indices = np.linspace(0, N-1, nb_control, dtype = 'int')
    CONTROL = NEW[indices,:]

    t = np.linspace(0, 1, nb_points)
    KNOT = bs.Knotvector(CONTROL, degree)
    BSPLINE, DER1, DER2, DER3 = bs.BSplines_RoutinePython(CONTROL, KNOT, degree, t, dimension = 3)

    return CONTROL, BSPLINE

############################################################################################
################################### MESH IMPROVEMENT WITH TRIMESH ##########################
############################################################################################

def Mesh_Improvement_Trimesh(file_marching_cube, name_export_mesh, coeff_smooth = 0.4):

    print(" ---> Marching_Cube Mesh read by Trimesh")
    aorte = trimesh.load_mesh(file_marching_cube)
    print(" ---> Number of original vertices : {}".format(len(aorte.vertices)))
    print(" ---> Number of original faces : {}".format(len(aorte.faces)))

    print(" ---> Smoothing the Mesh with a Laplacian Filter of coeff : {}".format(coeff_smooth))
    smooth = trimesh.smoothing.filter_laplacian(aorte, lamb = coeff_smooth)

    print(" ---> Remeshing by Subdivision")
    sub = smooth.subdivide()
    print(" ---> Number of new vertices : {}".format(len(sub.vertices)))
    print(" ---> Number of new faces : {}".format(len(sub.faces)))

    print(" ---> Mesh watertight ? {}. If False fill holes.".format(sub.is_watertight))
    if sub.is_watertight == False :
        sub.fill_holes()

    print(" ---> Fixing normals")
    sub.fix_normals()

    sub.export(name_export_mesh)

    return 0

############################################################################################
################################### MESH SLICE #############################################
############################################################################################

def Mesh_Slice(file_mesh_closed, PARAMETRIZATION, path_writer):

    aorte = trimesh.load_mesh(file_mesh_closed)

    SLICE_1_COORD = PARAMETRIZATION[0,0:3]
    SLICE_1_TAN = PARAMETRIZATION[0,3:6]

    SLICE_2_COORD = PARAMETRIZATION[-1,0:3]
    SLICE_2_TAN = PARAMETRIZATION[-1,3:6]

    slice_1 = aorte.slice_plane(SLICE_1_COORD, SLICE_1_TAN)
    slice_2 = slice_1.slice_plane(SLICE_2_COORD, -SLICE_2_TAN)

    slice_2.export(path_writer)

    return 0

############################################################################################
################################### MESH GENERATION FROM PCL  ##############################
############################################################################################


def Mesh_generation(CONTOUR, path_writer, nb_centerline, nb_thetas):

    print("Calcul du maillage")
    cell = []
    for k in range(nb_centerline - 1):
        P0 = CONTOUR[ k*nb_thetas,:][np.newaxis]
        T1 = CONTOUR[ (k+1)*nb_thetas:(k+2)*nb_thetas,:]
        decal = np.argmin(distance.cdist(P0,T1))
        for i in range(nb_thetas - 1) :
            cell_1 = np.array([ (k*nb_thetas)+i , (k+1)*nb_thetas + (i+decal)%(nb_thetas-1) , (k+1)*nb_thetas + (i+1+decal)%(nb_thetas-1)])
            cell.append(cell_1)
            cell_2 = np.array([ (k*nb_thetas)+i , (k*nb_thetas)+i+1 , (k+1)*nb_thetas + (i+1+decal)%(nb_thetas-1)])
            cell.append(cell_2)
    faces = [("triangle", np.vstack(cell))]
    mesh = meshio.Mesh(CONTOUR, faces)
    print("Ecriture sous format .stl de nom : mesh_initial.stl")
    meshio.write(path_writer, mesh)#, file_format = "stl")

    print("Lecture du maillage par trimesh")
    aorte = trimesh.load_mesh(path_writer)
    print("Quelques réparations sur le maillage")
    trimesh.repair.fix_normals(aorte)
    trimesh.repair.fix_inversion(aorte)
    trimesh.repair.fix_winding(aorte)
    print("Calcul des éléments du maillage")
    nb_vertices = len(aorte.vertices)
    nb_triangles = len(aorte.faces)
    print("Nombre de points = ", nb_vertices)
    print("Nombre de triangles = ", nb_triangles)
    print("Export du maillage au format .stl sous le nom : new_mesh.stl")
    os.remove(path_writer)
    aorte.export(path_writer)
