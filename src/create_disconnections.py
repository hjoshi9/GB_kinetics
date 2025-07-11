import numpy as np
import os
import subprocess

from src.min_shuffle import *
from src.lammps import *
from src.disconnection_generation import *
from src.solid_angle_calculations import *
from src.read_write_lammpsdatafiles import *

def generate_disconnection_images(gb_data,bur,step_height,lat_par,lat_Vec,axis,sizes,size_z,elem,reg_parameter,max_iters,lammps_location,mpi_location,folder,potential,dispy,dispz):
    """
    Generates disconnection images, calls the functions that are required

    Parameters
    ----------
    gb_data : 1D array
        Vector containing important GB information like sigma,mis,inc,period.
    bur : float
        Burgers vector of disconnection mode.
    step_height : float
        Step height of disconnection mode.
    lat_par : float
        Lattice parameter.
    lat_Vec : 2D array
        Primitive lattice vectors for crystal system.
    axis : 1D array
        Tilt axis.
    sizes : int
        Factor that controls box size along the GB. Box length along GB = 2*CSL_period*size.
    elem : string
        Element under consideration.
    reg_parameter : float
        Regularization parameter for skinhorn algorithm.
    max_iters : float
        Maximum number of iterations for skinhorn algorithm.
    lammps_location : string
        Location of lmp_serial and lmp_mpi executables.
    mpi_location : string
        Location of mpirun exeutable.
    folder : string
        Input folder.
    potential : string
        Location and name of LAMMPS potential to be used.
    dispy : float
        Displacement along gb.
    dispz : float
        displacement along axis.

    Returns
    -------
    out_folder : string
        Output folder.
        
    """
    # GB properties
    sigma = int(gb_data[0])
    mis = np.round(gb_data[1])
    inc = gb_data[2]
    p = gb_data[3]*lat_par
    non_per_cutoff = p*6
    size = sizes
    size_factor = 20
    if size == 1:
        size_factor = 40
    nCells = size_factor*size
    inc = gb_data[2]
    A,B,ga,gb = create_bicrystal(gb_data, axis, lat_par, lat_Vec)
    offset = -0*p/2
    gb_pos = 0
    section_factor = 2
    dipole_number = 3
    create_bicrystal_decision = True # or False
    min_decision = True# or False
    min_shuffle_decision = True # or False
    create_eco_input =  True
    out_folder = folder + elem+"/Sigma"+str(int(sigma))+"/Misorientation"+str(np.round(mis))+"/size"+str(size)+"/b"+str(np.round(bur,2))+"h"+str(np.round(step_height,2))+"/"
    os.makedirs(out_folder,exist_ok = True)
    total_images = int(sizes*section_factor)+1
    for image_num in range(total_images):
        if image_num < total_images:
            nodes = np.array([[-image_num*p/section_factor+offset,step_height+gb_pos],[image_num*p/section_factor+offset,step_height+gb_pos]])
        else:
            fac = 4
            nodes = np.array([[-image_num*p/section_factor-p/2*fac,step_height+gb_pos],[image_num*p/section_factor+p/2*fac,step_height+gb_pos]])
        disloc1 = nodes[0]
        disloc2 = nodes[1]
        nImages = 2
        if image_num == 0:
            if create_bicrystal_decision == True:
                print("\n============= Generating initial GB bicrystallographically ==================")
                gA,gB,box1 = generate_disconnection_ordered_initial(A,B,nCells,p,axis,lat_par,non_per_cutoff,bur,size,size_z,nodes,nImages,step_height,gb_pos)
                gA_1 = gA
                gB_1 = gB
                name_suffix = "size_"+str(size)+"disc"+str(image_num)
                file_name_init,box = write_lammps_input_ordered(out_folder,gA,gB,elem,sigma,inc,axis,2,box1,name_suffix,gb_pos,0,disloc1[0],disloc2[0])
                #print(out_folder + file_name_init)
            if min_decision == True:
                print("========================= Minimizing using LAMMPS ===========================")
                min_outputfile_init = "data."+elem+"s"+str(sigma)+"_size"+str(size)+"_min_d"+str(image_num)
                f,fout = write_minimization_input(elem,sigma,mis,inc,lat_par,size,out_folder,file_name_init,min_outputfile_init,dispy,dispz,potential)
                if min_shuffle_decision == True:
                    command = lammps_location + "/lmp_serial -in " + f
                else:
                    command = mpi_location + "/mpirun -np 6 "+ lammps_location + "/lmp_mpi -in" + f
                subprocess.run([command],shell=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("Done writing minimized file " + fout)
            #break
        else:
            if create_bicrystal_decision == True:
                if image_num<total_images:
                    print("\n================= Generating GB image " + str(image_num) + " bicrystallographically ==============")
                    gA,gB,box1 = generate_disconnection_ordered_other(A,B,nCells,p,axis,lat_par,non_per_cutoff,bur,size,size_z,nodes,nImages,step_height,gA_1,gB_1,gb_pos,image_num,dipole_number)
                    name_suffix = "size_"+str(size)+"disc"+str(image_num)
                    file_name1,box = write_lammps_input_ordered(out_folder,gA,gB,elem,sigma,inc,axis,2,box1,name_suffix,gb_pos,step_height,disloc1[0],disloc2[0])
                    #print(out_folder + file_name1)
                else:
                    print("\n================ Generating final GB bicrystallographically =================")
                    gA,gB,box1 = generate_disconnection_ordered_final(A,B,nCells,p,axis,lat_par,non_per_cutoff,bur,size,size_z,nodes,nImages,step_height,gb_pos)
                    name_suffix = "size_"+str(size)+"disc"+str(image_num)
                    file_name1,box = write_lammps_input_ordered(out_folder,gA,gB,elem,sigma,inc,axis,2,box1,name_suffix,gb_pos,step_height,disloc1[0],disloc2[0])
                    #print(out_folder + file_name1)
            
            # minimize
            if min_decision == True:
                print("========================== Minimizing using LAMMPS ==========================")
                min_outputfile = "data."+elem+"s"+str(sigma)+"_size"+str(size)+"_min_d"+str(image_num)
                f,fout = write_minimization_input(elem,sigma,mis,inc,lat_par,size,out_folder,file_name1,min_outputfile,dispy,dispz,potential)
                if min_shuffle_decision == True:
                    command = lammps_location + "/lmp_serial -in " + f
                else:
                    command = mpi_location + "/mpirun -np 6 "+ lammps_location + "/lmp_mpi -in" + f
                subprocess.run([command],shell=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) 
                print("Done writing minimized file " + fout)
                
            if min_shuffle_decision == True:
                print("==================== Generating atomic trajectories =========================")
                # apply min shuffle
                file_mode = 2
                file = min_outputfile_init
                data_init = read_LAMMPS_datafile(out_folder+file,file_mode)
                file = min_outputfile
                data_final = read_LAMMPS_datafile(out_folder+file,file_mode)
                
                
                box = data_final[0][2]
                filepath = out_folder+"data."+elem+"s"+str(sigma)+"inc0.0_size_"+str(size)+"disc"+str(0)+"_minmov"
                #data.Cus13inc0.0_size_2disc0_minmov
                average_gb_loc,gb_hi,gb_lo = find_gb_location(filepath)
                if step_height > 0:
                    gb_loc = 0*average_gb_loc-0.5*lat_par*p/16
                else:
                    gb_loc = 1*average_gb_loc+0.5*lat_par*p/16
                h = step_height + 1*step_height 
                d_start = disloc1[0]-0.5*5
                d_stop = disloc2[0]+0.5*5
                version = "_step" + str(image_num)
                
                atoms = data_init[0][3]
                Ai = []
                Bi = []
                neb_init = []
                for i in range(atoms.shape[0]):
                    if atoms[i,1] == 1.0:
                       Ai.append([atoms[i,2],atoms[i,3],atoms[i,4],atoms[i,0]])
                     #elif atoms
                    elif atoms[i,1] == 2.0:
                       Bi.append([atoms[i,2],atoms[i,3],atoms[i,4],atoms[i,0]]) 
                    else:
                       neb_init.append([atoms[i,2],atoms[i,3],atoms[i,4],atoms[i,0]]) 
                Ai = np.asarray(Ai)
                Bi = np.asarray(Bi)
                initial = atoms
                
                atoms = data_final[0][3]
                Af = []
                Bf = []
                neb_final = []
                for i in range(atoms.shape[0]):
                    if atoms[i,1] == 1.0:
                        Af.append([atoms[i,2],atoms[i,3],atoms[i,4],atoms[i,0]])
                    #elif atoms[i,2] == 
                    elif atoms[i,1] == 2.0:
                        Bf.append([atoms[i,2],atoms[i,3],atoms[i,4],atoms[i,0]]) 
                    else:
                        neb_final.append([atoms[i,2],atoms[i,3],atoms[i,4],atoms[i,0]]) 
                Af = np.asarray(Af)
                Bf = np.asarray(Bf)
                final = atoms
                
                Ai_n = np.array(neb_init)
                Bf_n = np.array(neb_final)
                if step_height>0:
                    neb_file =  min_shuffle_input(Ai,Bi,Af,Bf,gb_loc,h,box,out_folder,sigma,inc,elem,d_start,d_stop)
                else:
                    neb_file =  min_shuffle_input_negative(Ai,Bi,Af,Bf,gb_loc,h,box,out_folder,sigma,inc,elem,d_start,d_stop)
                
                Xs,Ys,Dvec,indicies,gamma_mod = min_shuffle(3, lat_par, sigma, mis,out_folder,elem,reg_parameter,max_iters,image_num)
                
                if step_height>0:
                    final_file = neb_structure_minimized(Ai,Bi,Xs,Ys,Af,Bf,box,step_height,bur,out_folder,sigma,inc,elem,2,size,d_start,d_stop,version)
                else:
                    final_file = neb_structure_minimized_negative(Ai,Bi,Xs,Ys,Af,Bf,box,h,bur,out_folder,sigma,inc,elem,2,size,d_start,d_stop,version,initial,final)
                if min_shuffle_decision == True:
                    file = "data.Cus"+str(sigma)+"inc0.0__step"+str(image_num)
                    outfile = "data.Cus"+str(sigma)+"inc0.0_out_step"+str(image_num)
                    f = write_neb_input_file(out_folder,file,image_num,outfile)
    #%% 
    if create_eco_input ==  True:
        fff = create_fix_eco_orientationfile(int(sigma), np.round(mis), inc, out_folder, A, B,lat_par)
    return  out_folder
