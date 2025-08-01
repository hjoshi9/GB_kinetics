import numpy as np
from src.GBKineticsRunController import runGridSearch


#---------- Input parameters for GB in question ---------------#
# Element
element = "Cu"
# Sigma value of Gb under consideration
sigma = 13
# Misorientation of the Gb
misorientation = 67.4
# Inclination of the GB
inclination = 0.0
# Lattice parameter of the element (try using the lat par corresponding to the potential you intend to use)
lattice_parameter = 3.615
# Tilt axis of GB
axis = [0, 0, 1]
# Size of system along the GB period in terms of 2*CSL period
size_y = 4
# Size of system along the tilt axis in terms of 2*CSL period
size_z = 2
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
# Run
results_folder_path = runGridSearch(sigma, misorientation, inclination, lattice_parameter,
                                    lattice_vector, axis, size_y,size_z, element,lammps_location,
                                    mpi_location, output_folder, lammps_potential, oilab_output_file)