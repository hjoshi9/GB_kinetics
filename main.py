import numpy as np
import os
from src.create_disconnections import generate_disconnection_images
from src.generate_gb_information import gb_props
from src.run_neb import run_neb_calc


# Input parameters for GB in question
element = "Cu"                                   # Element 
sigma = 13                                       # Sigma value of Gb under consideration
misorientation = 22.6                            # Misorientation of the Gb
lattice_parameter = 3.615                        # Lattice parameter of the element (try using the lat par corresponding to the potential you intend to use)
axis = [0,0,1]                                   # Tilt axis of GB
size_y = 2                                       # Size of system along the GB period in terms of 2*CSL period
size_z = 4.5                                       # Size of system along the tilt axis in terms of 2*CSL period
lattice_vector = np.array([[0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.5],
                    [0.5, 0.0, 0.5]])            # Lattice Vectors for the crystal system (current implementation is tested for fcc only)

# Parameters for min-shuffle algorithm
reg = 0.05                                       # Regularization parameter for min-shuffle algorithm
iterMax = 50000                                  # Maximum iterations for min-shuffle algorithm
               
choose_disconnection  = 1                        # Put 1 if you want to choose the disconnection mode

# Locations of programs needed for running the program
lammps_location = "/opt/homebrew/bin"            # Location of directory where lmp_serial and lmp_mpi are stored
mpi_location = "/opt/homebrew/bin"               # Location of directory where mpirun is stored
output_folder = "output/"                        # Location of directory where output is to be stored, 
                                                 # the program creates subdirectories for each element, sigma value , misorientation and disconnection mode within it   
lammps_potential = "/opt/homebrew/Cellar/lammps/20240829-update1/share/lammps/potentials/Cu_mishin1.eam.alloy" # Full path to the potential to be used

# Displacements obtained from running grid_search script
disp_along_gb = 0.0
disp_along_tilt = -0.9

# Parameters for neb run on lammps
partitions = 40                                  # Number of partitions used for neb calculations
mode = 1                                         # Variable lets you choose if you want to run intermediate images through NEB or not. mode = 1 -> NEB with intermediate images, mode = 0 -> NEb with just the initial and final GB images

# Run
gb_data,burgers_vector,step_height = gb_props(sigma,misorientation,axis,lattice_parameter,choose_disconnection)
results_folder_path = generate_disconnection_images(gb_data,burgers_vector,step_height,lattice_parameter,lattice_vector,axis,size_y,size_z,element,reg,iterMax,lammps_location,mpi_location,output_folder,lammps_potential,disp_along_gb,disp_along_tilt)
#neb = run_neb_calc(results_folder_path,element,lattice_parameter,sigma,misorientation,size,burgers_vector,step_height,partitions,mode,lammps_potential,mpi_location,lammps_location)
