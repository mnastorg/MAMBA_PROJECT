# Project description

## Overview of Anaconda

The codes above are all written in Python with use of data-science platform _Anaconda_ : https://www.anaconda.com

Anaconda and its conda distribution can be installed following this link : https://docs.anaconda.com/anaconda/install/

The Conda distribution is a very useful tool to manage Python packages and create shareable environments. More precisely, if one wants to make a project, he will use a certain number of packages (numpy, scipy ... for the most common ones) which are not necessarly installed in your conda distribution. If the project's creator wants to share his work, the receiver needs to have those packages to properly run the codes. In that case, the creator can build an environment specific to his project (which contains all required packages) and share it to his co-workers. 

We recommand first to get familiar with those notions by reading : 
- Get Started : https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
- How to manage environments : https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#

The codes we are going to use all along this project are compiling with Python version : 3.6.1. (If you decide to create an environment for this work with a higher python version then some packages won't download and you couldn't run the codes !)

## The Mamba environment ? 

Run in a Terminal window the following lines : 

1. Create the environment from the mamba_macos.yml file : **conda env create -f environment.yml**
2. Activate the new environment : **conda activate mamba_macos**
3. Verify that the new environment was installed correctly : **conda info --envs**

As you can see on the .yml file name, we built the environment on a Mac Os operating systems, which can (and probably will) provides errors while doing step 1 if you use another system. In that case, I recommand taking a look at the following link : https://johannesgiorgis.com/sharing-conda-environments-across-different-operating-systems/.

If you still can't build the environment the following packages are the ones to install manually (with the creation of a python environment with version **3.6.1**) : geomdl // importlib // ipython // itk // matplotlib // meshio // networkx // numpy // openmesh-python // openssl // pandas // pyglet // python 3.6.1 // scikit-fmm // scikit-learn // scipy // symfit // sympy // trimesh // vmtk // pip // vtk. (All the packages can be read in the .yml files in details).

## How to run codes ? 

Codes are usually organized in the following order : The __main__ files which contains the principal functions and the other Python files that are located in the different folders. 

If you don't want to change anything in the code, then you only have to run those __main__ files. To do it, we use the python prompt **ipython** but you can also use Spyder or any other Python file interpreter. We use Atom to vizualise and edit Python files. 

The basic commands are the following : 

1. Open a Terminal window and go the project's folder.
2. Activate mamba_macos environment : **conda activate mamba_macos**
3. Open ipython : **ipython**
4. Then you have to run the __main__ file that you're interested in. For instance, if you want to use fitting program : **run main_fitting**
5. Once you did that you only have to launch the function that interest you with the corresponding correct arguments.

## How to vizualise files ?

Most of the files can be read in Paraview : https://www.paraview.org 

1. STL / VTP / VTU ... files are directly read once uploaded in Paraview.
2. CSV files are readable after selecting Filters --> Alphabetical --> Table to points then select in properties "x, y, z" and colors depending on other scalars.
3. For nii.gz files (3d Segmented images) it is possible to use free open-source software ITK-SNAP.

# What can you do ? 

## 3D mathematical skeleton fitting

## Aortic aneurysm wall parametrization

## 


