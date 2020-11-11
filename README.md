# Overview of Conda distribution

The codes above are all written in Python with use of data-science platform _Anaconda_ : https://www.anaconda.com

Anaconda and its conda distribution can be installed following this link : https://docs.anaconda.com/anaconda/install/

The Conda distribution is a very useful tool to manage Python packages and create shareable environments. More precisely, if one wants to make a project, he will use a certain number of packages (numpy, scipy ... for the most common ones) which are not necessarly installed in your conda distribution. If the project's creator wants to share his work, the receiver needs to have those packages to properly run the codes. In that case, the creator can build an environment specific to his project (which contains all required packages) and share it to his co-worker. 

We recommand first to get familiar with those notions by reading : 
- Get Started : https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
- How to manage environments : https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#

The codes we are going to use all along this project are compiling with Python version : 3.6.1. (If you decide to create an environment for this work with a higher python version then some packages won't download and you couldn't run the codes !)

# How to build the Mamba environment ? 

