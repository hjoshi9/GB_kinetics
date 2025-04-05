import numpy as np
import os
import subprocess

from src.min_shuffle import *
from src.lammps import *
from src.oilab import *
from src.disconnection_generation import *
from src.solid_angle_calculations import *
from src.write_neb_read_structures import *
from src.read_write_lammpsdatafiles import *

def generate_disconnection_images(gb_data,bur,step_height,lat_par,lat_Vec,axis,sizes,elem,reg_parameter,max_iters):
    
    # GB properties
    folder = "output/"
    sigma = gb_data[0]
    mis = gb_data[1]
    inc = gb_data[2]
    p = gb_data[3]*lat_par
    non_per_cutoff = p*6
    size = 2
    size_factor = 20
    if size == 1:
        size_factor = 25
    nCells = size_factor*size
    inc = gb_data[2]
    A,B,ga,gb = create_bicrystal(gb_data, axis, lat_par, lat_Vec)
    offset = 0  # works for m = -1
    gb_pos = 0
    section_factor = 1
    dipole_number = 3
    create_bicrystal_decision = True # or False
    min_decision = True# or False
    min_shuffle_decision = True # or False
    create_eco_input =  True
    for image_num in range(sizes+1):
        if image_num < sizes:
            nodes = np.array([[-image_num*p/section_factor+offset,step_height+gb_pos],[image_num*p/section_factor+offset,step_height+gb_pos]])
        else:
            fac = 0
            nodes = np.array([[-image_num*p/section_factor+p/2*fac,step_height+gb_pos],[image_num*p/section_factor+p/2*fac,step_height+gb_pos]])
        disloc1 = nodes[0]
        disloc2 = nodes[1]
        nImages = 2
        if image_num == 0:
            if create_bicrystal_decision == True:
                gA,gB,box1 = generate_disconnection_ordered_initial(A,B,nCells,p,axis,lat_par,non_per_cutoff,bur,size,nodes,nImages,step_height,gb_pos)
                gA_1 = gA
                gB_1 = gB
                out_folder = folder + elem+"/Solid_angle/Sigma"+str(sigma)+"_new/Misorientation"+str(np.round(mis))+"/size"+str(size)+"/b"+str(np.round(bur,2))+"h"+str(np.round(step_height,2))+"/"
                os.makedirs(out_folder,exist_ok = True)
                
                # Write data
                name_suffix = "size_"+str(size)+"disc"+str(image_num)
                file_name_init,box = write_lammps_input_ordered(out_folder,gA,gB,elem,sigma,inc,axis,2,box1,name_suffix,gb_pos,0,disloc1[0],disloc2[0])
                print(out_folder + file_name_init)
            
            # minimize
            # For m = 1
            dispy = 0.0
            dispz = -0.9
            # For m = 2
            #dispy = -0.3
            #dispz = 0.5
            if min_decision == True:
                print("Minimizing using LAMMPS \n")
                min_outputfile_init = "data."+elem+"s"+str(sigma)+"_size"+str(size)+"_min_d"+str(image_num)
                f = write_minimization_input(elem,sigma,mis,inc,lat_par,size,out_folder,file_name_init,min_outputfile_init,dispy,dispz)
                if min_shuffle_decision == True:
                    command = "/opt/homebrew/bin/lmp_serial -in " + f
                else:
                    command = "/opt/homebrew/bin/mpirun -np 6 /opt/homebrew/bin/lmp_mpi -in " + f
                subprocess.run([command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,shell=True)
            #break
        else:
            #conitnue
            # minimize
            # For m = 1
            dispy = 0.0 #0.45 (m = 2, n=-1) , 0.9 (m = 1)
            dispz = -0.9
            # For m = 2
            #dispy = -0.3
            #dispz = 0.5
            if create_bicrystal_decision == True:
                if image_num<=sizes:
                    gA,gB,box1 = generate_disconnection_ordered_other(A,B,nCells,p,axis,lat_par,non_per_cutoff,bur,size,nodes,nImages,step_height,gA_1,gB_1,gb_pos,image_num,dipole_number)
                    # Set up the folder that we would write the data into
                    out_folder = folder + elem+"/Solid_angle/Sigma"+str(sigma)+"_new/Misorientation"+str(np.round(mis))+"/size"+str(size)+"/b"+str(np.round(bur,2))+"h"+str(np.round(step_height,2))+"/"
                    os.makedirs(out_folder,exist_ok = True)
                    
                    # Write data
                    name_suffix = "size_"+str(size)+"disc"+str(image_num)
                    file_name1,box = write_lammps_input_ordered(out_folder,gA,gB,elem,sigma,inc,axis,2,box1,name_suffix,gb_pos,step_height,disloc1[0],disloc2[0])
                    print(out_folder + file_name1)
                else:
                    gA,gB,box1 = generate_disconnection_ordered_final(A,B,nCells,p,axis,lat_par,non_per_cutoff,bur,size,nodes,nImages,step_height,gb_pos)
                    out_folder = folder + elem+"/Solid_angle/Sigma"+str(sigma)+"/Misorientation"+str(np.round(mis))+"/size"+str(size)+"/b"+str(np.round(bur,2))+"h"+str(np.round(step_height,2))+"/"
                    os.makedirs(out_folder,exist_ok = True)
                    
                    # Write data
                    name_suffix = "size_"+str(size)+"disc"+str(image_num)
                    file_name1,box = write_lammps_input_ordered(out_folder,gA,gB,elem,sigma,inc,axis,2,box1,name_suffix,gb_pos,step_height,disloc1[0],disloc2[0])
                    print(out_folder + file_name1)
            
            # minimize
            if min_decision == True:
                print("Minimizing using LAMMPS \n\n\n")
                min_outputfile = "data."+elem+"s"+str(sigma)+"_size"+str(size)+"_min_d"+str(image_num)
                f = write_minimization_input(elem,sigma,mis,inc,lat_par,size,out_folder,file_name1,min_outputfile,dispy,dispz)
                if min_shuffle_decision == True:
                    command = "/opt/homebrew/bin/lmp_serial -in " + f
                else:
                    command = "/opt/homebrew/bin/mpirun -np 6 /opt/homebrew/bin/lmp_mpi -in " + f
                subprocess.run([command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,shell=True) 
            
            if min_shuffle_decision == True:
                # apply min shuffle
                file_mode = 2
                file = min_outputfile_init
                data_init = read_LAMMPS_datafile(out_folder+file,file_mode)
                file = min_outputfile
                data_final = read_LAMMPS_datafile(out_folder+file,file_mode)
                
                
                box = data_final[0][2]
                gb_loc = -0.5*lat_par*p/32#+ 0.2*math.copysign(1,step_height) #-1  #- 0.3
                h = step_height + 0.25*step_height # 1*step_height (m = 2. n= -1) #*math.copysign(1,step_height)#+1     #-1.2+    #+0.8
                d_start = disloc1[0]-0.5*0
                d_stop = disloc2[0]+0.5*0
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
                    #neb_file =  min_shuffle_input(Ai,Bi,Af,Bf,gb_loc,h,box,folder,sigma,inc,elem,d_start,d_stop)
                    neb_file =  min_shuffle_input_negative(Ai,Bi,Af,Bf,gb_loc,h,box,out_folder,sigma,inc,elem,d_start,d_stop)
                
                Xs,Ys,Dvec,indicies,gamma_mod = min_shuffle(3, lat_par, sigma, mis,out_folder,elem,reg_parameter,max_iters,image_num)
                image_num = 1
                if step_height>0:
                    #final_file = neb_structure_minimized_negative(Ai,Bi,Xs,Ys,Af,Bf,box,h,burgers,out_folder,sigma,inc,elem,2,size,d_start,d_stop,version,initial,final)
                    final_file = neb_structure_minimized(Ai,Bi,Xs,Ys,Af,Bf,box,step_height,bur,out_folder,sigma,inc,elem,2,size,d_start,d_stop,version)
                else:
                    final_file = neb_structure_minimized_negative(Ai,Bi,Xs,Ys,Af,Bf,box,h,bur,out_folder,sigma,inc,elem,2,size,d_start,d_stop,version,initial,final)
    #%% 
    if min_shuffle_decision == True:
        for i in range(sizes+1):
            file = "data.Cus"+str(sigma)+"inc0.0__step"+str(i)
            outfile = "data.Cus"+str(sigma)+"inc0.0_out_step"+str(i)
            f = write_neb_input_file(out_folder,file,image_num,outfile)
    #%% 
    if create_eco_input ==  True:
        eco_folder = "output/Cu/Solid_angle/Sigma"+str(sigma)+"/Misorientation"+str(mis)+"/"
        fff = create_fix_eco_orientationfile(int(sigma), np.round(mis), inc, eco_folder, A, B,lat_par)
        
        
    return 0