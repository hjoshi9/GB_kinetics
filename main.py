import numpy as np

from src.create_disconnections import *
from src.generate_gb_information import *

element = "Cu"                                   # Element 
sigma = 13                                       # Sigma value of Gb under consideration
misorientation = 22.62                           # Misorientation of the Gb
lattice_parameter = 3.615                        # Lattice parameter of the element (try using the lat par corresponding to the potential you intend to use)
axis = [0,0,1]                                   # Tilt axis of GB
size = 2                                         # Size of system along the GB period in terms of 2*CSL period
lattice_vector = np.array([[0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.5],
                    [0.5, 0.0, 0.5]])            # Lattice Vectors for the crystal system (current implementation is tested for fcc only)
reg = 0.01                                       # Regularization parameter for min-shuffle algorithm
iterMax = 500000                                 # Maximum iterations for min-shuffle algorithm
               
# Figure out disconnection information
gb_data,burgers_vector,step_height = gb_props(sigma,misorientation,axis,lattice_parameter)

# Create disconnection images
lammps_location = "/opt/homebrew/bin" # Location of directory where lmp_serial and lmp_mpi are stored
mpi_location = "/opt/homebrew/bin"    # Location of directory where mpirun is stored
output_folder = "output/"             # Location of directory where output is to be stored, 
                                      # the program creates subdirectories for each element, sigma value , misorientation and disconnection mode within it   
lammps_potential = "/opt/homebrew/Cellar/lammps/20240829-update1/share/lammps/potentials/Cu_mishin1.eam.alloy" # Full path to the potential to be used
disp_along_gb = 0.0
disp_along_tilt = -0.9
a = generate_disconnection_images(gb_data,burgers_vector,step_height,lattice_parameter,lattice_vector,axis,size,element,reg,iterMax,lammps_location,mpi_location,output_folder,lammps_potential,disp_along_gb,disp_along_tilt)

# Run neb calculations on the images generated

