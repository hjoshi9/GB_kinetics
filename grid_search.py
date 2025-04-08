import numpy as np
import subprocess
import os

from src.lammps import write_lammps_gridsearch_input
from src.disconnection_generation import generate_disconnection_ordered_initial,create_bicrystal
from src.generate_gb_information import gb_props
from src.read_write_lammpsdatafiles import write_lammps_input_ordered


element = "Cu"                                   # Element 
sigma = 13                                       # Sigma value of Gb under consideration
misorientation = 22.62                           # Misorientation of the Gb
lattice_parameter = 3.615                        # Lattice parameter of the element (try using the lat par corresponding to the potential you intend to use)
axis = [0,0,1]                                   # Tilt axis of GB
size = 2                                         # Size of system along the GB period in terms of 2*CSL period
lattice_vector = np.array([[0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.5],
                    [0.5, 0.0, 0.5]])            # Lattice Vectors for the crystal system

lammps_location = "/opt/homebrew/bin" # Location of directory where lmp_serial and lmp_mpi are stored
mpi_location = "/opt/homebrew/bin"    # Location of directory where mpirun is stored
output_folder = "output/"             # Location of directory where output is to be stored, 
                                      # the program creates subdirectories for each element, sigma value , misorientation and disconnection mode within it   
lammps_potential = "/opt/homebrew/Cellar/lammps/20240829-update1/share/lammps/potentials/Cu_mishin1.eam.alloy" # Full path to the potential to be used

# Create GB
mode = 0	
gb_data,burgers_vector,step_height = gb_props(sigma,misorientation,axis,lattice_parameter,mode)
A,B,ga,gb = create_bicrystal(gb_data, axis, lattice_parameter, lattice_vector)
nCells = 50
p = gb_data[3]*lattice_parameter
non_per_cutoff = p*10
gb_pos = 0
nodes = np.array([[0,0],[1,0]])
disloc1 = nodes[0]
disloc2 = nodes[1]
nImages = 1
gA,gB,box1 = generate_disconnection_ordered_initial(A,B,nCells,p,axis,lattice_parameter,non_per_cutoff,burgers_vector,size,nodes,nImages,step_height,gb_pos)
name_suffix = "size_"+str(size)+"perfect_gb"
out_folder = "output/gridsearch/Sigma"+str(sigma)+"_misorientation"+str(int(gb_data[1]))+"/"
os.makedirs(out_folder,exist_ok = True)
file_name_init,box = write_lammps_input_ordered(out_folder,gA,gB,element,sigma,gb_data[2],axis,2,box1,name_suffix,gb_pos,0,disloc1[0],disloc2[0])
limit = 1
step_increments = 0
mode = 1
output_file,input_file = write_lammps_gridsearch_input(out_folder,file_name_init,out_folder,lammps_potential,element,lattice_parameter,int(gb_data[0]),size,int(gb_data[1]),step_increments,limit,mode)
command = mpi_location + "/mpirun -np 6 "+ lammps_location + "/lmp_mpi -in " + input_file
subprocess.run([command],stderr=subprocess.DEVNULL,shell=True) 
