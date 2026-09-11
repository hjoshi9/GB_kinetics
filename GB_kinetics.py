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
latticeParameter = 3.615
# Tilt axis of GB
axis = [0, 0, 1]
# Size of system along the GB period in terms of 2*CSL period
size_along_gb_period = 4
# Size of system along the tilt axis in terms of 2*CSL period
size__along_tilt_axis = 2
# Lattice Vectors for the crystal system (current implementation is tested for fcc only)
lattice_vector = np.array([[0.5, 0.5, 0.0],
                           [0.0, 0.5, 0.5],
                           [0.5, 0.0,0.5]])

#-------------- Parameters for min-shuffle algorithm --------------#
# Regularization parameter for min-shuffle algorithm.
# Raised automatically (doubled, up to six times) if Sinkhorn will not converge at it
# within maximumIterations; the value actually used is printed for every solve.
regularizationParameter = 0.005
# Maximum iterations for min-shuffle algorithm, per regularization attempt.
maximumIterations = 1000000
# Solve the shuffle once on the smallest bicrystal there is -- one CSL period long,
# one repeat thick -- and map that pattern onto every image of the real system,
# instead of solving optimal transport separately for each one.
#
# Behind a disconnection the boundary has simply moved: one slab of grain A has become
# grain B, identically everywhere inside the loop, so the whole shuffle fits in one CSL
# cell. Solving it per image re-derives that cell over and over, at a cost that grows
# as the square of the atoms involved -- which is why large systems are slow.
#
# Set False to solve every image from scratch (the original behaviour).
map_shuffle_from_reference = True
# Go further and build each image by replicating the relaxed reference cell, rather
# than masking it out of the reference lattice. The elementary shuffle and the
# disconnection's plastic field are laid over the tiled cell. Turning this on turns
# map_shuffle_from_reference on too, since both need the same reference.
#
# Off by default, because it only pays when minimization is reduced. Measured on
# Sigma17 at size 2, minimizing every image as this script does:
#
#   image          0     1     2     3     4   total
#   constructed  802   717   593   894   526    3532  CG steps
#   replicated    22   878   923  1080  1040    3943  CG steps
#
# The flat GB is a true repeat of the reference and converges 36x faster. Every other
# image carries two disconnection cores and a plastic field that varies across it, so
# a relaxed cell with that field laid over it is no closer to the minimum than ideal
# sites are -- it is slightly further. Both routes relax to the same energy to within
# 1e-5 eV/atom, so this is a question of cost, not correctness.
#
# Set True together with minimize_endpoints_only to get the benefit: then the
# 2*size_along_gb_period-1 intermediate minimizations are skipped outright.
replicate_from_reference = True
# Minimize only the flat GB and the fully stepped boundary, leaving the images in
# between as they were built. Those two are the states whose energies mean something;
# the images between them are a starting path, and NEB relaxes it.
#
# Only worth turning on alongside replicate_from_reference, which builds those
# intermediates out of an already relaxed cell. Without it they are ideal lattice
# sites under an analytic slip field, and handing NEB that as a path is a much rougher
# start than this saves.
minimize_endpoints_only = True
# Put True if you want to choose the disconnection mode, False will automatically create disconnection mode with
# smallest burgers vector and corresponding step height
chooseDisconnection = True

# ------------------------- Input and output folders -------------------- #
# Location of bicrystallographic data obtained using oILAB
oilab_output_file = "data/fcc0-10.txt"
# Location of directory where output is to be stored, the program creates subdirectories
# for each element, sigma value , misorientation and disconnection mode within it
output_folder = "output/"

# ---------- Location of programs needed to run this (change these) ----------------------#
# Location of the directory holding the LAMMPS binaries, and of the one holding mpirun.
# Leave both as None to have them discovered automatically: PATH and the usual install
# directories are searched, and any build lacking the styles this pipeline needs
# (eam/alloy for minimization, neb for NEB) is skipped rather than used and left to fail
# mid-run.  Set them to a directory to search it first, or export GBK_LMP_SERIAL /
# GBK_LMP_MPI / GBK_MPIRUN to name the exact binaries.
# Run `python -m src.executables` to see what discovery picks on this machine.
lammps_location = None
# Location of directory where mpirun is stored
mpi_location = None
# Full path to the potential to be used
lammps_potential = "/home/himanshu/Desktop/mylammps/potentials/Cu_mishin1.eam.alloy"

# Displacements obtained from running grid_search script
disp_along_gb = 0.0
disp_along_tilt = -0.9

# Parameters for neb run on lammps
# Number of partitions used for neb calculations
partitions = 4
# Variable lets you choose if you want to run intermediate images through NEB or not.
# mode = 1 -> NEB with intermediate images, mode = 0 -> NEb with just the initial and final GB images
neb_mode = 1
# Variable which allows for switching off automatically triggering neb calculations (in case you only need disconnection images)
run_neb = False
# Which alternative shuffle chain to run NEB on. Every chain is written to its own
# branch<N> subfolder regardless; this only picks the one NEB is deployed to.
# 0 is the heaviest at every image, and the only chain when regularizationParameter = 0.
neb_branch = 0

# Run
if __name__ == "__main__":
    results_folder_path = runGBkinetics(sigma,
                                        misorientation,
                                        inclination,
                                        latticeParameter,
                                        lattice_vector,
                                        axis,
                                        size_along_gb_period,
                                        size__along_tilt_axis,
                                        element,
                                        regularizationParameter,
                                        maximumIterations,
                                        lammps_location,
                                        mpi_location,
                                        output_folder,
                                        lammps_potential,
                                        disp_along_gb,
                                        disp_along_tilt,
                                        oilab_output_file,
                                        chooseDisconnection,
                                        run_neb,
                                        neb_mode,
                                        partitions,
                                        neb_branch,
                                        map_shuffle_from_reference,
                                        replicate_from_reference,
                                        minimize_endpoints_only)