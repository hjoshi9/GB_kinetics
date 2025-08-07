import numpy as np
from src.GBKineticsRunController import *

#---------- Input parameters for GB in question ---------------#
# Element
element = "Cu"
# Sigma value of Gb under consideration
sigma = 17
# Misorientation of the Gb
misorientation = 28
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

#-------------- Parameters for min-shuffle algorithm --------------#
# Regularization parameter for min-shuffle algorithm
reg = 0.005
# Maximum iterations for min-shuffle algorithm
iterMax = 1000000
# Put True if you want to choose the disconnection mode, False will automatically create disconnection mode with
# smallest burgers vector and corresponding step height
choose_disconnection = True

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

# Displacements obtained from running grid_search script
disp_along_gb = 0.0
disp_along_tilt = -0.9

# Parameters for neb run on lammps
# Number of partitions used for neb calculations
partitions = 40
# Variable lets you choose if you want to run intermediate images through NEB or not.
# mode = 1 -> NEB with intermediate images, mode = 0 -> NEb with just the initial and final GB images
neb_mode = 0
# Variable which allows for switching off automatically triggering neb calculations (in case you only need disconnection images)
run_neb = True
# Run
results_folder_path = runGBkinetics(sigma,
                                    misorientation,
                                    inclination,
                                    lattice_parameter,
                                    lattice_vector,
                                    axis,
                                    size_y,
                                    size_z,
                                    element,
                                    reg,
                                    iterMax,
                                    lammps_location,
                                    mpi_location,
                                    output_folder,
                                    lammps_potential,
                                    disp_along_gb,
                                    disp_along_tilt,
                                    oilab_output_file,
                                    choose_disconnection,
                                    run_neb)