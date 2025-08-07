import numpy as np
import os
from src.GBKineticsRunController import *


#---------- Input parameters for GB in question ---------------#
# Element
element = "Cu"
# Sigma value of Gb under consideration
sigma = 17
# Misorientation of the Gb
misorientation = 28.0
# Inclination of the GB
inclination = 0.0
# Lattice parameter of the element (try using the lat par corresponding to the potential you intend to use)
lattice_parameter = 3.615
# Tilt axis of GB
axis = [0, 0, 1]
# Size of system along the GB period in terms of 2*CSL period
size_y = 1
# Size of system along the tilt axis in terms of 2*CSL period
size_z = 1
# Lattice Vectors for the crystal system (current implementation is tested for fcc only)
lattice_vector = np.array([[0.5, 0.5, 0.0],
                           [0.0, 0.5, 0.5],
                           [0.5, 0.0,0.5]])

# ------------- Input and output folders (do not change) -------------------- #
# Location of bicrystallographic data obtained using oILAB
oilab_output_file = "data/fcc0-10.txt"
# Location of directory where output is to be stored, the program creates subdirectories
# for each element, sigma value , misorientation and disconnection mode within it
output_folder = "output/"

# ---------- Location of programs needed to run this (change these) ----------------------#
# Location of directory where lmp_serial and lmp_mpi are stored
lammps_location = "/opt/homebrew/bin"
# Location of directory where mpirun is stored
mpi_location = "/opt/homebrew/bin"
# Full path to the potential to be used
lammps_potential = "/opt/homebrew/Cellar/lammps/20240829-update1/share/lammps/potentials/Cu_mishin1.eam.alloy"

# Number of cores to be used for grid search simulation
num_cores = os.cpu_count()-2
# Increments in displacements to be used for grid search
# Defines how fine you want the grid to be for the grid search
step_increments = 1
# Highest displacement to be used for grid search
# Defines how large of a space you want to explore in terms of Angstroms
limit = 1
# Setting that decides if you want dump files to be created along with the text output
# 1 --> Create dump files that show GB configs
# 0 --> Does not create dump files, only the text file that contains displacements and GB energy
output_setting = 1
# Setting that allows for user to choose which disconnection mode is to be used
# Keep this to False as grid search is done for flat GB anyway
choose_disconnection = False
# Run
results_folder_path = runGridSearch(sigma,
                                    misorientation,
                                    inclination,
                                    lattice_parameter,
                                    lattice_vector,
                                    axis,
                                    size_y,
                                    size_z,
                                    element,
                                    lammps_location,
                                    mpi_location,
                                    output_folder,
                                    lammps_potential,
                                    oilab_output_file,
                                    choose_disconnection,
                                    num_cores,
                                    step_increments,
                                    limit,
                                    output_setting)