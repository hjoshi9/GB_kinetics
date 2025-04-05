import numpy as np
from src.read_write_lammpsdatafiles import read_LAMMPS_datafile

def neb_structure_minimized(g_A,g_B,Xs,Ys,Af,Bf,box,step_height,burgers_vector,folder,sigma,inc,elem,mode,y_size,disc_start,disc_end,version):
    """
    Write neb read atom positions for a bicrystal for a disconnection mode of positive step height
    """
    suffix = ["_step0",version]
    for ii in range(len(suffix)):
        file = "data." + elem + "s" +str(sigma) + "inc" + str(inc) + "_" +suffix[ii]
        name = folder + file
        natoms = g_A.shape[0]+g_B.shape[0]
        f = open(name,"w")
        eps = 0.1
        #f.write("# LAMMPS data file Sigma = %d, inclination = %f\n"%(sigma,inc))
        f.write("#LAMMPS data file\n")
        f.write("%d atoms\n"%(natoms))
        f.write("2 atom types\n")
        f.write("%0.10f %0.10f xlo xhi\n"%(box[0,0]-eps,box[0,1]+eps))
        f.write("%0.10f %0.10f ylo yhi\n"%(box[1,0],box[1,1]))
        f.write("%0.10f %0.10f zlo zhi\n"%(box[2,0],box[2,1]))
        f.write("0.0 0.0 0.0 xy xz yz\n\n")
        f.write("Atoms # atomic\n\n")
        k= 1
        g_A2 = g_A
        g_B2 = g_B
        if suffix[ii]=="_step0":
            grain_A = []
            grain_B = []
            for i in range(g_A.shape[0]):
                grain_num = 1
                for j in range(len(Xs)):
                    if g_A[i,3]==Xs[j,3]:
                        g_A[i,:] = Xs[j,:]
                        #print(g_A[i,:])
                        grain_num = 1
                f.write("%d %d %0.10f %0.10f %0.10f\n"% (g_A[i,3],grain_num,g_A[i,0],g_A[i,1],g_A[i,2]))
                grain_A.append([g_A[i,0],g_A[i,1],g_A[i,2],g_A[i,3]])
                k += 1
            print(k)
            for i in range(g_B.shape[0]):
                grain_num = 2
                #f.write("%d\t%d\t%0.6f\t%0.6f\t%0.6f\n"%(k,grain_num,g_B[0,i],g_B[1,i],g_B[2,i]))
                f.write("%d %d %0.10f %0.10f %0.10f\n"% (g_B[i,3],grain_num,g_B[i,0],g_B[i,1],g_B[i,2]))
                grain_B.append([g_B[i,0],g_B[i,1],g_B[i,2],g_B[i,3]])
                k += 1
            print(k)
            f.close()
        else:
            
            grain_A2 = []
            grain_B2 = []
            for i in range((g_A2.shape[0])):
                grain_num = 1
                flag = 0
                for j in range(len(Ys)):
                    if abs(g_A2[i,3]-Xs[j,3])<0.1:
                        g_A2[i,:] = Ys[j,:]
                        #print(Ys[j,:])
                        grain_num = 2
                        flag = 1
                if flag == 0:
                    for ii in range(Af.shape[0]):
                        if abs(g_A2[i,3]-Af[ii,3])<0.1:
                            g_A2[i,:] = Af[ii,:]
                f.write("%d %d %0.10f %0.10f %0.10f\n"% (g_A2[i,3],grain_num,g_A2[i,0],g_A2[i,1],g_A2[i,2]))
                grain_A2.append([g_A[i,0],g_A[i,1],g_A[i,2],g_A[i,3]])
                k += 1
            print(k)
            for i in range(g_B2.shape[0]):
                grain_num = 2
                flag = 0
                for j in range(len(Ys)):
                    if abs(g_B2[i,3]-Xs[j,3])<0.1:
                        g_B2[i,:] = Ys[j,:]
                        #print(Ys[j,:])
                        grain_num = 2
                        flag = 1
                if flag == 0:
                    for ii in range(Bf.shape[0]):
                        if abs(g_B2[i,3]-Bf[ii,3])<0.1:
                            g_B2[i,:] = Bf[ii,:]
                f.write("%d %d %0.10f %0.10f %0.10f\n"% (g_B2[i,3],grain_num,g_B2[i,0],g_B2[i,1],g_B2[i,2]))
                grain_B2.append([g_B2[i,0],g_B2[i,1],g_B2[i,2],g_B2[i,3]])
                k += 1
            print(k)
            f.close()
            
            # Create the array
            
    print("Done writing bicrystal")
    print(name)
    return file,np.array(grain_A),np.array(grain_B)


def neb_structure_minimized_negative(g_A,g_B,Xs,Ys,Af,Bf,box,step_height,burgers_vector,folder,sigma,inc,elem,mode,y_size,disc_start,disc_end,version,initial,final):
    """
    Write neb read atom positions for a bicrystal for a disconnection mode of negative step height
    """
    suffix = ["_step0",version]
    #suffix = [version]
    for ii in range(len(suffix)):
        file = "data." + elem + "s" +str(sigma) + "inc" + str(inc) + "_" +suffix[ii]
        name = folder + file
        natoms = g_A.shape[0]+g_B.shape[0]
        k= 1
        print(g_A.shape[0],g_B.shape[0])
        g_A2 = g_A
        g_B2 = g_B
        if suffix[ii]=="_step0":
            f = open(name,"w")
            eps = 0.1
            #f.write("# LAMMPS data file Sigma = %d, inclination = %f\n"%(sigma,inc))
            f.write("#LAMMPS data file\n")
            f.write("%d atoms\n"%(natoms))
            f.write("2 atom types\n")
            f.write("%0.10f %0.10f xlo xhi\n"%(box[0,0]-eps,box[0,1]+eps))
            f.write("%0.10f %0.10f ylo yhi\n"%(box[1,0],box[1,1]))
            f.write("%0.10f %0.10f zlo zhi\n"%(box[2,0],box[2,1]))
            f.write("0.0 0.0 0.0 xy xz yz\n\n")
            f.write("Atoms # atomic\n\n")
            grain_A = []
            grain_B = []
            '''
            for i in range(g_A.shape[0]):
                grain_num = 1
                f.write("%d %d %0.10f %0.10f %0.10f\n"% (g_A[i,3],grain_num,g_A[i,0],g_A[i,1],g_A[i,2]))
                grain_A.append([g_A[i,0],g_A[i,1],g_A[i,2],g_A[i,3]])
                k += 1
            for i in range(g_B.shape[0]):
                grain_num = 2
                #f.write("%d\t%d\t%0.6f\t%0.6f\t%0.6f\n"%(k,grain_num,g_B[0,i],g_B[1,i],g_B[2,i]))
                f.write("%d %d %0.10f %0.10f %0.10f\n"% (g_B[i,3],grain_num,g_B[i,0],g_B[i,1],g_B[i,2]))
                grain_B.append([g_B[i,0],g_B[i,1],g_B[i,2],g_B[i,3]])
                k += 1
            '''
            for i in range(len(initial)):
                f.write("%d %d %0.10f %0.10f %0.10f\n"% (initial[i,0],initial[i,1],initial[i,2],initial[i,3],initial[i,4]))
            f.close()
            print("Done writing bicrystal")
            print(name)
        else:
            f = open(name,"w")
            eps = 0.1
            #f.write("# LAMMPS data file Sigma = %d, inclination = %f\n"%(sigma,inc))
            f.write("#LAMMPS data file\n")
            f.write("%d atoms\n"%(natoms))
            f.write("2 atom types\n")
            f.write("%0.10f %0.10f xlo xhi\n"%(box[0,0]-eps,box[0,1]+eps))
            f.write("%0.10f %0.10f ylo yhi\n"%(box[1,0],box[1,1]))
            f.write("%0.10f %0.10f zlo zhi\n"%(box[2,0],box[2,1]))
            f.write("0.0 0.0 0.0 xy xz yz\n\n")
            f.write("Atoms # atomic\n\n")
            grain_A2 = []
            grain_B2 = []
            count = 0
            '''
            for i in range((g_A2.shape[0])):
                grain_num = 1
                flag = 0
                if flag == 0:
                    for ii in range(Af.shape[0]):
                        if abs(g_A2[i,3]-Af[ii,3])<0.1:
                            g_A2[i,:] = Af[ii,:]
                f.write("%d %d %0.10f %0.10f %0.10f\n"% (g_A2[i,3],grain_num,g_A2[i,0],g_A2[i,1],g_A2[i,2]))
                grain_A2.append([g_A[i,0],g_A[i,1],g_A[i,2],g_A[i,3]])
                k += 1
            for i in range(g_B2.shape[0]):
                grain_num = 2
                flag = 0
                for j in range(len(Ys)):
                    if abs(g_B2[i,3]-Xs[j,3])<0.1:
                        g_B2[i,:] = Ys[j,:]
                        #print(Ys[j,:])
                        grain_num = 1
                        count +=1
                        flag = 1
                if flag==0:
                    for ii in range(Bf.shape[0]):
                        if abs(g_B2[i,3]-Bf[ii,3])<0.1:
                            g_B2[i,:] = Bf[ii,:]
                f.write("%d %d %0.10f %0.10f %0.10f\n"% (g_B2[i,3],grain_num,g_B2[i,0],g_B2[i,1],g_B2[i,2]))
                grain_B2.append([g_B2[i,0],g_B2[i,1],g_B2[i,2],g_B2[i,3]])
                k += 1
            '''
            for i in range(len(initial)):
                out = initial[i,:]
                flag = 0
                for j in range(len(Xs)):
                    grain_num = 1
                    if abs(initial[i,0]-Xs[j,3])<0.1:
                        out = np.array([Ys[j,3],grain_num,Ys[j,0],Ys[j,1],Ys[j,2]])
                        count +=1
                        flag = 1
                if flag==0:
                    for k in range(len(final)):
                        if abs(initial[i,0]-final[k,0])<0.1:
                            out = final[k,:]
                #print(flag,out-final[k,:])
                f.write("%d %d %0.10f %0.10f %0.10f\n"% (out[0],out[1],out[2],out[3],out[4]))
            f.close()
            print(count,len(Xs),len(Ys))
            f.close()
            
            print("Done writing bicrystal")
            print(name)
    
    return file

def write_neb_input_file(folder,file,image_num,outfile):
    in_file = folder + file
    out_file = folder + outfile
    data = read_LAMMPS_datafile(in_file,1)
    natoms = data[0][0]
    atoms = data[0][3]
    f = open(out_file,"w")
    
    f.write("%d\n"%(natoms))
    for i in range(len(atoms)):
        f.write("%d %f %f %f\n"%(atoms[i,0],atoms[i,2],atoms[i,3],atoms[i,4]))
    f.close()
    return out_file
