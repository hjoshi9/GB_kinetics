import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.interpolate import griddata
from src.IO import *

class run_LAMMPS:
    """
        A class to set up, run, and post-process LAMMPS simulations
        for grain boundary disconnection energy calculations.

        Attributes:
            folder (str): Directory for simulation input/output files.
            element (str): Chemical element symbol (e.g. 'Cu').
            lattice_parameter (float): Lattice parameter for the element.
            sigma (int): Sigma value for grain boundary.
            misorientation (float): Misorientation angle.
            inclination (float): Grain boundary inclination.
            size_along_gb_period (int): Size along grain boundary periodicity.
            potential (str): Path to interatomic potential file.
            mpi_location (str): Path to MPI binaries.
            lammps_location (str): Path to LAMMPS binaries.
            output_filename_gridsearch (str or None): Filename for grid search output.
            gridsearch_output_setting (int or None): Grid search output setting flag.
            gridsearch_output_folder (str or None): Folder for grid search outputs.
            lammps_input_filename (str or None): Filename of the current LAMMPS input script.
            neb_images (int or None): Number of NEB images (partitions).
            number_of_images_provided (int or None): Number of images in NEB calculation.
            burgers_vec (float or None): Burgers vector magnitude.
            step_height (float or None): Step height used in NEB calculation.
            neb_output_folder (str or None): Folder for NEB output files.
    """
    def __init__(self, folder, element, lattice_parameter, sigma, misorientation,
                 inclination, size_along_gb_period,
                  potential, mpi_location, lammps_location):
        """
            Initializes the run_LAMMPS instance with simulation parameters.

            Args:
                folder (str): Directory to store simulation files.
                element (str): Chemical element symbol.
                lattice_parameter (float): Lattice parameter.
                sigma (int): Sigma value for grain boundary.
                misorientation (float): Misorientation angle in degrees.
                inclination (float): Grain boundary inclination angle.
                size_along_gb_period (int): Size along grain boundary period.
                potential (str): Path to interatomic potential file.
                mpi_location (str): Path to MPI binaries.
                lammps_location (str): Path to LAMMPS binaries.
         """
        self.folder = folder
        self.element = element
        self.lattice_parameter = lattice_parameter
        self.sigma = sigma
        self.misorientation = misorientation
        self.inclination = inclination
        self.size_along_gb_period = size_along_gb_period

        self.potential = potential
        self.mpi_location = mpi_location
        self.lammps_location = lammps_location

        self.output_filename_gridsearch = None
        self.gridsearch_output_setting = None
        self.gridsearch_output_folder = None

        self.lammps_input_filename = None
        self.neb_images = None
        self.number_of_images_provided = None
        self.burgers_vec = None
        self.step_height = None
        self.neb_output_folder = None

    def write_minimization_input(self, file_name, dispy, dispz,minimization_along_gb_directions=False):
        """
            Writes a LAMMPS input script for energy minimization with optional box relaxation.

            Args:
                file_name (str): Base filename for input/output.
                dispy (float): Displacement along grain boundary periodic direction.
                dispz (float): Displacement along tilt axis.
                minimization_along_gb_directions (bool): If True, performs additional minimization
                    steps with box relaxation along y and z directions.
        """
        sigma = self.sigma
        mis   = self.misorientation
        inc   = self.inclination
        lat_par = self.lattice_parameter
        size = self.size_along_gb_period
        potential = self.potential
        elem = self.element
        folder = self.folder

        min_file = "min.in"
        file = os.path.join(folder, min_file)
        self.lammps_input_filename = file
        min_outputfile = file_name + "_min"
        min_output_movie = file_name + "_minmov"
        self.output_filename = min_outputfile
        images = 2 * size + 1
        if minimization_along_gb_directions:
            extra_minimization_script = f"""
#-------- Minimize with box relaxation in y direction -------------
fix 1 all box/relax y 0 vmax 0.001
min_style cg
minimize 1e-25 1e-25 5000 5000

#------- Minimize with box relaxation in z direction ---------------
fix 1 all box/relax z 0 vmax 0.001
min_style cg
minimize 1e-25 1e-25 5000 5000

#--------------------- Minimize ------------------------------------
min_style cg
minimize 1e-25 1e-25 10000 10000
            """
        else:
            extra_minimization_script = "\n"

        lammps_script = f"""# Minimization of Sigma{sigma} disconnections using LAMMPS
#------------------General settings for simulation----------------------------
clear
units metal
dimension 3
boundary m p p
atom_style atomic
neighbor 0.3 bin
neigh_modify delay 5
atom_modify map array sort 0 0.0

#----------------- Variable declaration---------------------------------------
variable a equal {lat_par}
variable potpath string {potential}
variable sigma equal {sigma}
variable y_image equal {size}
variable dispy equal {dispy}
variable dispz equal {dispz}
variable elem string {elem}
variable mis equal {mis}
variable folder string {folder}
variable file_name string ${{folder}}{file_name}
variable out_file1 string ${{folder}}{min_outputfile}
variable out_file_mov string ${{folder}}{min_output_movie}

#----------------------- Atomic structure ----------------------
lattice fcc $a
read_data ${{file_name}}
group upper type 1
group lower type 2

#----------------------- InterAtomic Potential --------------------
pair_style eam/alloy
pair_coeff * * ${{potpath}} {elem} {elem}
neighbor 2 bin
neigh_modify delay 10 check yes

#-------------------- Define compute settings ------------------------
compute csym all centro/atom fcc
compute energy all pe/atom
compute eng all reduce sum c_energy

#---------- Displace top part for lowest energy structure ----------
delete_atoms overlap 1 upper upper
delete_atoms overlap 1 lower lower
delete_atoms overlap 0.1 upper lower
displace_atoms upper move 0 ${{dispy}} ${{dispz}} units box

# Apply fix to tether centroid of the system to the center

#--------------------- Minimize ------------------------------------
thermo 250
thermo_style custom step temp pe lx ly lz press pxx pyy pzz
dump 1 all custom 100 ${{out_file_mov}} id type x y z c_csym c_energy
min_style cg
minimize 1e-25 1e-25 10000 10000
{extra_minimization_script}
write_data ${{out_file1}}
        """

        with open(file, "w") as f:
            f.write(lammps_script)

        outfile = os.path.join(folder, min_outputfile) #folder + min_outputfile

    def write_lammps_gridsearch_input(self, infile, outfolder,step_increments, limit, output_setting=0):
        """
            Writes a LAMMPS input script for performing a grid search over displacements.

            Args:
                infile (str): Name of the input data file.
                outfolder (str): Output folder to store results.
                step_increments (float): Incremental displacement step size.
                limit (float): Maximum displacement limit in both directions.
                output_setting (int): Flag to enable detailed output; 1 enables additional dumps.
        """
        sigma = self.sigma
        mis = self.misorientation
        inc = self.inclination
        lat_par = self.lattice_parameter
        size = self.size_along_gb_period
        potential = self.potential
        elem = self.element
        folder = self.folder

        f_name = "grid_search.in"
        file = os.path.join(folder, f_name)
        self.lammps_input_filename = file
        outfile = elem + "_" + "sigma" + str(sigma) + "_mis" + str(mis) + "_size" + str(size) + "_gridsearch_results.txt"
        self.output_filename_gridsearch = outfile
        yloop = int(2 * limit/step_increments) + 1
        zloop = yloop
        yloop0 = int(limit/step_increments) + 1
        zloop0 = yloop0
        dispy_expr = f"{step_increments}*(${{disp_county}}-{yloop0})"
        dispz_expr = f"{step_increments}*(${{disp_countz}}-{zloop0})"

        outfile2_line = ""
        dump_line = ""
        outfolder2_line = ""
        if output_setting == 1:
            outfolder_configs = os.path.join(outfolder, "/configs")
            os.makedirs(outfolder_configs, exist_ok=True)
            outfolder2_line = f"variable outfolder_configs string {outfolder_configs}"
            outfile2_line = f"variable outfile2 string ${{outfolder_configs}}/{infile}_dy${{dispy}}dz${{dispz}}\n"
            dump_line = "dump            1 all custom 5000 ${outfile2} id type x y z c_csym c_eng\n"

        script = f"""# LAMMPS script to run a grid search
variable disp_county loop {yloop}
label loopy
variable disp_countz loop {zloop}
label loopz
variable a equal {lat_par}
variable potpath string {potential}
variable sigma equal {sigma}
variable y_image equal {size}
variable dispy equal {dispy_expr}
variable dispz equal {dispz_expr}
variable step equal 0
variable elem string {elem}
variable mis equal {mis}
variable folder string {outfolder}
variable file_name string ${{folder}}/{infile}
{outfolder2_line}
{outfile2_line}
variable outfile string ${{folder}}/{outfile}

clear
units metal
dimension 3
boundary m p p
atom_style atomic
echo both
neighbor        0.3 bin
neigh_modify    delay 1
atom_modify  map array sort 0 0.0
lattice fcc $a
read_data ${{file_name}}
replicate 1 1 2
group upper type 1
group lower type 2
variable midbox equal xhi/2+xlo/2
variable gb_thickness equal 10
variable gblo equal ${{midbox}}-${{gb_thickness}}
variable gbhi equal ${{midbox}}+${{gb_thickness}}
variable bulklo equal ${{midbox}}+1.2*${{gb_thickness}}
variable bulkhi equal xhi-0.25*${{gb_thickness}}
region BULK block ${{bulklo}} ${{bulkhi}} INF INF INF INF units box
region GB block ${{gblo}} ${{gbhi}} INF INF INF INF units box
variable area equal ly*lz
pair_style eam/alloy
pair_coeff * * ${{potpath}} {elem} {elem}
neighbor 2 bin
neigh_modify delay 10 check yes
compute eng all pe/atom
compute eatoms all reduce sum c_eng
compute csym all centro/atom fcc
group GB region GB
group BULK region BULK
compute peratom GB pe/atom
compute peratombulk BULK pe/atom
compute pe GB reduce sum c_peratom
compute pebulk BULK reduce sum c_peratombulk
variable peGB equal c_pe
variable peBULK equal c_pebulk
variable atomsGB equal count(GB)
variable atomsBULK equal count(BULK)
delete_atoms overlap 1 upper upper
delete_atoms overlap 1 lower lower
delete_atoms overlap 0.1 upper lower
displace_atoms upper move 0 ${{dispy}} ${{dispz}} units box
thermo 500
thermo_style custom step temp pe lx ly lz press pxx pyy pzz c_eatoms
{dump_line}
variable pot_eng equal pe

min_style cg
minimize 1e-25 1e-25 5000 5000

variable coh equal (${{peBULK}}/${{atomsBULK}})
variable conversion_factor equal 16.02
variable GBene equal (${{conversion_factor}}*(${{peGB}}/${{area}}-${{coh}}*${{atomsGB}}/${{area}}))
print "dispy = ${{dispy}} dispz = ${{dispz}} GBene = ${{GBene}}" append ${{outfile}}

next disp_countz
jump SELF loopz
next disp_county
jump SELF loopy
        """
        with open(file, "w") as f:
            f.write(script)

    def write_lammps_neb_input_script(self, burgers_vector, step_height,num_steps, partitions, mode=1):
        """
            Writes a LAMMPS input script for Nudged Elastic Band (NEB) calculations.

            Args:
                burgers_vector (float): Burgers vector magnitude for disconnection.
                step_height (float): Step height for the disconnection mode.
                num_steps (int): Number of NEB images (steps).
                partitions (int): Number of parallel partitions for NEB.
                mode (int): Mode flag controlling file naming and step count behavior.

            Returns:
                str: Path to the output folder for NEB calculation results.
        """
        folder = self.folder
        elem = self.element
        lat_par = self.lattice_parameter
        sigma = self.sigma
        mis = self.misorientation
        size = self.size_along_gb_period
        potential = self.potential
        b = burgers_vector
        self.burgers_vec = b
        h = step_height
        self.step_height = h
        neb_file = "neb.in"
        self.lammps_input_filename = neb_file
        file = os.path.join(folder, neb_file)
        output_folder = os.path.join(folder, "partitions" + str(partitions) + "/")
        os.makedirs(output_folder, exist_ok=True)
        if mode == 0:
            number_of_steps = 1
        else:
            number_of_steps = num_steps

        if mode == 0:
            final_file_var = "variable final_file string ${folder}/data.Cus${sigma}inc0.0_out_step${size}"
        else:
            final_file_var = "variable final_file string ${folder}/data.Cus${sigma}inc0.0_out_step${s}"

        script = f"""\
        # LAMMPS NEB input for Sigma = {sigma}; misorientation = {mis}; disconnection mode (b,h) = ({b},{h}); size = {size}
        variable s loop {number_of_steps}
        label step_loop
        variable a equal {lat_par}
        variable elem string {elem}
        variable potpath string {potential}
        variable sigma equal {sigma}
        variable mis equal {mis}
        variable size equal {size}
        variable folder string {folder}
        variable outfolder string {output_folder}
        variable step_number equal $s-1
        variable b equal {b}
        variable h equal {h}
        variable initial_file string ${{folder}}/data.Cus${{sigma}}inc0.0__step${{step_number}}
        {final_file_var}
        clear
        units metal
        boundary m p p
        atom_style atomic
        neighbor        0.3 bin
        neigh_modify    delay 1
        atom_modify  map array sort 0 0.0
        echo both
        lattice fcc $a
        read_data ${{initial_file}}
        group upper type 1
        group lower type 2
        pair_style eam/alloy
        pair_coeff * * ${{potpath}} {elem} {elem}
        neighbor 2 bin
        neigh_modify delay 10 check yes
        compute csym all centro/atom fcc
        compute eng all pe/atom
        region nearGB block INF INF INF INF INF INF units box
        group nebatoms region nearGB
        group nonnebatoms subtract all nebatoms
        timestep 0.01
        fix 1b nebatoms neb 1.0 perp 1.0 parallel neigh
        thermo 10
        variable u uloop 100
        min_style quickmin
        dump 1 all custom 2000 ${{outfolder}}/dump.neb_${{elem}}sigma${{sigma}}size${{size}}discb${{b}}h${{h}}_step${{s}}.$u id type x y z c_eng c_csym
        neb 0.0 0.0 10000 10000 1000 final ${{final_file}}
        next s
        jump SELF step_loop
        """

        with open(file, "w") as f:
            f.write(script)
        return output_folder

    def run_neb_calc(self, burgers_vector, step_height,number_of_steps, partitions=40, mode=1):
        """
            Runs the NEB calculation using MPI-parallelized LAMMPS.

            Args:
                burgers_vector (float): Burgers vector magnitude.
                step_height (float): Step height.
                number_of_steps (int): Number of NEB steps.
                partitions (int, optional): Number of parallel partitions (default: 40).
                mode (int, optional): Mode flag for NEB input script generation (default: 1).

            Raises:
                RuntimeError: If the NEB LAMMPS run fails.
        """
        mpi_location = self.mpi_location
        lammps_location = self.lammps_location
        folder = self.folder
        print("\n========================== Running neb calculations ==============================")
        neb_output_folder = self.write_lammps_neb_input_script(burgers_vector, step_height,number_of_steps, partitions, mode)
        print("The results from this calculation will be stored in " + neb_output_folder)
        command = mpi_location + "/mpirun --oversubscribe --use-hwthread-cpus -np " + str(
            partitions) + " " + lammps_location + "/lmp_mpi  -partition " + str(partitions) + "x1 -in " + folder+self.lammps_input_filename
        #subprocess.run([command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        result = subprocess.run([command], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(
                f"NEB run failed:\nSTDOUT:\n{result.stdout.decode()}\nSTDERR:\n{result.stderr.decode()}")
        self._remove_temp_files(neb_output_folder,"neb")
        print("Done with NEB calculations!!")
        self.neb_images = partitions
        self.number_of_images_provided = number_of_steps
        self.neb_output_folder = neb_output_folder

    def run_minimization(self,file_name,disp_along_gb_period,disp_along_tilt_axis,minimization_along_gb_directions=False):
        """
            Runs a minimization calculation using LAMMPS.

            Args:
                file_name (str): Base filename for input/output files.
                disp_along_gb_period (float): Displacement along grain boundary period.
                disp_along_tilt_axis (float): Displacement along tilt axis.
                minimization_along_gb_directions (bool, optional): Whether to perform box relaxations (default: False).

            Returns:
                str: Filename of the minimized structure output.

            Raises:
                RuntimeError: If the minimization LAMMPS run fails.
        """
        lammps_location = self.lammps_location
        dispy = disp_along_gb_period
        dispz = disp_along_tilt_axis
        print("========================== Minimizing using LAMMPS ==========================")
        self.write_minimization_input(file_name, dispy, dispz,minimization_along_gb_directions)
        command = lammps_location + "/lmp_serial -in " + self.lammps_input_filename
        #subprocess.run([command], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        result = subprocess.run([command], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(
                f"LAMMPS minimization failed:\nSTDOUT:\n{result.stdout.decode()}\nSTDERR:\n{result.stderr.decode()}")
        print("Done writing minimized file " + self.output_filename)
        return self.output_filename

    def run_grid_search(self,infile, outfolder,number_of_cores = 6,step_increments=0.1,limit=1,output_setting=0):
        """
            Runs a grid search of displacements using MPI-parallelized LAMMPS.

            Args:
                infile (str): Input data filename.
                outfolder (str): Output folder for results.
                number_of_cores (int, optional): Number of MPI cores to use (default: 6).
                step_increments (float, optional): Step size for grid search increments (default: 0.1).
                limit (float, optional): Maximum displacement limit (default: 1).
                output_setting (int, optional): Output verbosity setting (default: 0).

            Returns:
                str: Filename of the grid search results.

            Raises:
                RuntimeError: If the grid search LAMMPS run fails.
        """
        print("\n=================================== Running grid search  ========================================")
        mpi_location = self.mpi_location
        lammps_location = self.lammps_location
        self.write_lammps_gridsearch_input(infile, outfolder,step_increments,limit, output_setting)
        command = mpi_location + "/mpirun -np " + str(number_of_cores)+ " " + lammps_location + "/lmp_mpi -in " + self.lammps_input_filename
        #subprocess.run([command], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        result = subprocess.run([command], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(
                f"Grid search run failed:\nSTDOUT:\n{result.stdout.decode()}\nSTDERR:\n{result.stderr.decode()}")
        print("Done with grid search!!")
        print("Results stored in : " +outfolder+ self.output_filename_gridsearch)
        self.gridsearch_output_setting = output_setting
        self.gridsearch_output_folder = outfolder
        self._remove_temp_files(outfolder)
        return self.output_filename_gridsearch

    def post_process_neb_output(self,outfolder,plot_decision=False):
        """
            Post-processes the NEB output dump files and optionally plots the energy profile.

            Args:
                outfolder (str): Folder to write processed output files.
                plot_decision (bool, optional): If True, generates and saves energy plot (default: False).
        """
        sigma = self.sigma
        size = self.size_along_gb_period
        partitions = self.neb_images
        steps = self.number_of_images_provided
        b = self.burgers_vec
        h = self.step_height
        folder = self.neb_output_folder
        mis = self.misorientation
        elem = self.element
        print("================================ Running neb post-processing ==================================")
        out_file = f"dump.neb_{elem}sigma{sigma}_mis{mis}_size{size}_discb{b}h{h}_partition{partitions}"
        message = "Writing output dump file: " + folder + out_file
        print(message)
        pot_eng = []
        for k in range(steps):
            for j in range(partitions):
                file = folder + f"dump.neb_{elem}sigma{sigma}size{size}discb{b}h{h}_step{k+1}.{j+1}"
                data = read_neb_output_data(file,2)
                tstep = partitions*k+j
                box = data[-1][2]
                Atoms = data[-1][3]
                natoms = Atoms.shape[0]
                pe = np.sum(Atoms[:,5])
                area = (box[1,1]-box[1,0])*(box[2,1]-box[2,0])/100 #nm^2
                pot_eng.append([tstep,pe,area])
                f = write_LAMMPSoutput_tstep(natoms,Atoms,box,outfolder,out_file,tstep)
        peng = np.array(pot_eng)

        out_name = outfolder + f"{elem}sigma{sigma}_mis{mis}_size{size}_discb{b}h{h}_partition{partitions}.txt"
        message = "Writing output dump file: " + out_name
        print(message)
        np.savetxt(out_name,peng)

        if plot_decision == True:
            # plotting
            plt.figure(dpi=300)
            plt.plot(peng[:,0]/len(peng),(peng[:,1]-peng[0,1])/peng[0,2])
            plt.grid()
            plt.xlabel("Reaction coordinate")
            plt.ylabel(r"$\Delta$E/a (eV/nm$^2$)")
            fig_name = outfolder + f"{elem}sigma{sigma}_mis{mis}_size{size}_discb{b}h{h}_partition{partitions}.png"
            message = "Writing output dump file: " + fig_name
            print(message)
            plt.savefig(fig_name)

    def post_process_gridsearch_data(self,plot_decision=False):
        """
            Post-processes grid search results, sorting and optionally plotting the GB energy surface.

            Args:
                plot_decision (bool, optional): If True, generates and saves contour plots (default: False).
        """
        sigma = self.sigma
        size = self.size_along_gb_period
        mis = self.misorientation
        elem = self.element
        outfolder = self.gridsearch_output_folder
        infile = self.output_filename_gridsearch
        print("=========================== Running grid search post-processing ==================================")
        gs_data = read_gridsearch_results(outfolder+infile)
        gs_data = gs_data[gs_data[:, 2].argsort()]
        outfile = outfolder + f"{elem}_sigma{sigma}_mis{mis}_size{size}_gridsearch_sorted.txt"
        write_gridsearch_results(outfile, gs_data)
        if plot_decision:
            # Create a regular grid to interpolate onto
            x = gs_data[:, 0]
            y = gs_data[:, 1]
            z = gs_data[:, 2]
            segments = len(x) * 2
            xi = np.linspace(np.min(x), np.max(x), segments)
            yi = np.linspace(np.min(y), np.max(y), segments)
            X, Y = np.meshgrid(xi, yi)
            # Interpolate z values onto the grid
            Z = griddata((x, y), z, (X, Y), method='cubic', rescale=True)
            # Plot
            fig = plt.figure(dpi=400, figsize=(8, 6))
            contour = plt.contourf(X, Y, Z, levels=segments, cmap='RdBu_r')
            plt.contour(X, Y, Z, levels=segments, colors='w', linewidths=0.00001)
            plt.xlabel("Displacement along gb period")
            plt.ylabel("Displacement along tilt axis")
            fig.colorbar(contour, location="right", orientation="vertical", shrink=1, pad=0.05, aspect=10,
                         label="GB energy")
            plt.scatter(x, y, c=z, s=10, cmap="RdBu_r")
            plt.tight_layout()
            outfile = outfolder + f"{elem}_sigma{sigma}_mis{mis}_size{size}_gridsearch_sorted.png"
            plt.savefig(outfile)

    @staticmethod
    def _remove_temp_files(output_folder,mode="non-neb"):
        """
            Moves the LAMMPS log file to output folder and removes temporary files.

            Args:
                output_folder (str): Folder where output files are stored.
                mode (str, optional): If "neb", removes additional NEB-related temp files (default: "non-neb").
        """
        # Move log file to results folder
        log_file = os.path.join(output_folder, "log.lammps")
        os.rename("log.lammps", log_file)
        if mode == "neb":
            # Remove temp files generated by lammps neb
            command = "rm screen.*"
            subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            command = "rm log.lammps*"
            subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            command = "rm tmp.lammps.variable"
            subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


