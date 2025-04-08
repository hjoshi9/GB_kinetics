import os
import subprocess
import numpy as np

def write_lammps_neb_input_script(folder,elem,lat_par,sigma,mis,size,b,h,partitions,mode,potential):
    """
    Writes lammps file to run neb calculations
    """
    neb_file = "neb.in"
    file = folder + neb_file
    output_folder = folder + "partitions"+str(partitions)+"/"
    os.makedirs(output_folder,exist_ok = True)
    if mode == 0:
        number_of_steps = 1
    else:
        number_of_steps = size
        
    f = open(file,"w")
    f.write("# LAMMPS NEB input for Sigma = %d; misorientation = %f; disconnection mode (b,h) = (%f,%f); size = %d\n"%(sigma,mis,b,h,size))
    f.write("variable s loop %d\n"%(number_of_steps))
    f.write("label step_loop\n")
    f.write("variable a equal %f\n"%(lat_par))
    f.write("variable elem string %s\n"%elem)
    f.write("variable potpath string %s\n"%(potential))
    f.write("variable sigma equal %d\n"%(sigma))
    f.write("variable mis equal %f\n"%(mis))
    f.write("variable size equal %d\n"%size)
    f.write("variable folder string %s\n"%(folder))
    f.write("variable outfolder string %s\n"%(output_folder))
    f.write("variable step_number equal $s-1\n")
    f.write("variable b equal %f\n"%b)
    f.write("variable h equal %f\n"%h)
    f.write("variable initial_file string ${folder}/data.Cus${sigma}inc0.0__step${step_number}\n")
    f.write("variable final_file string ${folder}/data.Cus${sigma}inc0.0_out_step${s}\n")
    f.write("clear\n")
    f.write("units metal\n")
    f.write("boundary m p p\n")
    f.write("atom_style atomic\n")
    f.write("neighbor        0.3 bin\n")
    f.write("neigh_modify    delay 1\n")
    f.write("atom_modify  map array sort 0 0.0\n")
    f.write("echo both\n")
    f.write("lattice fcc $a\n")
    f.write("read_data ${initial_file}\n")
    f.write("group upper type 1\n")
    f.write("group lower type 2\n")
    f.write("pair_style eam/alloy\n")
    f.write("pair_coeff * * ${potpath} %s %s\n"%(elem,elem))
    f.write("neighbor 2 bin\n")
    f.write("neigh_modify delay 10 check yes\n")
    f.write("compute csym all centro/atom fcc\n")
    f.write("compute eng all pe/atom\n")
    f.write("region nearGB block INF INF  INF INF INF INF units box\n")
    f.write("group nebatoms region nearGB\n")
    f.write("group nonnebatoms subtract all nebatoms\n")
    f.write("timestep 0.01\n")
    f.write("fix             1b nebatoms neb 1.0 perp 1.0 parallel neigh\n")
    f.write("thermo          10\n")
    f.write("variable       u uloop 100\n")
    f.write("min_style       quickmin\n")
    f.write("dump 1 all custom 2000 ${outfolder}/dump.neb_${elem}sigma${sigma}size${size}discb${b}h${h}_step${s}.$u id type x y z c_eng c_csym\n")
    f.write("neb               0.0 0.0 10000 10000 1000 final ${final_file}\n")
    f.write("next s\n")
    f.write("jump SELF step_loop\n")
    f.write("\n")
    f.close()
    return file,output_folder

def run_neb_calc(folder,elem,lat_par,sigma,mis,size,b,h,partitions,mode,potential,mpi_location,lammps_location):
    neb_input_file,neb_output_folder = write_lammps_neb_input_script(folder,elem,lat_par,sigma,np.round(mis),size,np.round(b,2),np.round(h,2),partitions,mode,potential)
    print("Starting neb calculations")
    print("The results from this calculation will be stored in "+neb_output_folder)
    command = mpi_location + "/mpirun --oversubscribe --use-hwthread-cpus -np "+ str(partitions) + " " + lammps_location + "/lmp_mpi  -partition "+ str(partitions)+"x1 -in " + neb_input_file
    subprocess.run([command],stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,shell=True)  # 
    return 0