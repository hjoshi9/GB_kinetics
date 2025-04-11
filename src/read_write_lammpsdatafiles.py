import numpy as np

def min_shuffle_input(Ai,Bi,Af,Bf,gb_loc,h,box,folder,sigma,inc,elem,disc_start,disc_end):
    """
    Reads input from lammps datafile which contains the information about the atoms on which minshuffle algorithm is to be acted on

    Parameters
    ----------
    Ai : 2D array
        Initial configuration of Grain 1.
    Bi : 2D array
        Initial configuration of Grain 2.
    Af : 2D array
        Final configuration of Grain 1.
    Bf : 2D array
        Final configuration of Grain 2.
    gb_loc : float
        GB location.
    h : float
        Step height of disconnection mode.
    box : 2D array
        Box dimensions.
    folder : string
        output folder.
    sigma : int
        Gb sigma value.
    inc : float
        GB inclination.
    elem : string
        Element.
    disc_start : float
        Disconnection start location.
    disc_end : float
        Disconnection end location.

    Returns
    -------
    A : 2D array
        Grain 1.
    B : 2D array
        Grain 2.

    """
    data_A = []
    data_B = []
    eps = 1e-1
    count = 1
    initial = np.concatenate((Ai,Bi),axis=0)
    for i in range(initial.shape[0]):
        if  initial[i,0]<gb_loc+h-eps and initial[i,0]>gb_loc-eps and initial[i,1]>=disc_start and initial[i,1]<= disc_end:
            count += 1
            data_A.append([initial[i,0],initial[i,1],initial[i,2],initial[i,3]])
    A = np.array(data_A)
    for i in range(Af.shape[0]):
        for j in range(len(data_A)):
            if abs(Af[i,3]-A[j,3])<0.5:
                count += 1
                data_B.append([Af[i,0],Af[i,1],Af[i,2],Af[i,3]])
                break
            
    for i in range(Bf.shape[0]):
        for j in range(len(data_A)):
            if abs(Bf[i,3]-A[j,3])<0.5:
                count += 1
                data_B.append([Bf[i,0],Bf[i,1],Bf[i,2],Bf[i,3]])
                break
            
    
    B = np.array(data_B)
    prefix = "min_shuffle"
    file = "data." + elem + "s" +str(sigma) + "inc" + str(inc) + "_" +prefix# + "_"+ str(mode)
    name = folder + file
    
    natoms = A.shape[0]+B.shape[0]
    #print("Check for same number of particles in the system")
    #print(A.shape[0],B.shape[0])
    f = open(name,"w")
    #f.write("# LAMMPS data file Sigma = %d, inclination = %f\n"%(sigma,inc))
    f.write("#LAMMPS data file\n")
    f.write("%d atoms\n"%(natoms))
    f.write("2 atom types\n")
    eps2 = 1
    xlo = gb_loc -h + eps2
    xhi = gb_loc + h + eps2
    mode = 2
    scale = 1.25
    f.write("%0.10f %0.10f xlo xhi\n"%(scale*xlo,scale*xhi))
    f.write("%0.10f %0.10f ylo yhi\n"%(box[1,0],box[1,1]))
    f.write("%0.10f %0.10f zlo zhi\n"%(box[2,0]+1e-2,box[2,1]))
    f.write("0.0 0.0 0.0 xy xz yz\n\n")
    f.write("Atoms # atomic\n\n")
    k= 1
    for i in range(A.shape[0]):
        grain_num = 1
        #f.write("%d\t%d\t%0.6f\t%0.6f\t%0.6f\n"% (k,grain_num,g_A[0,i],g_A[1,i],g_A[2,i]))
        f.write("%d %d %0.10f %0.10f %0.10f\n"% (A[i,3],grain_num,A[i,0],A[i,1],A[i,2]))
        k += 1
    for i in range(B.shape[0]):
        grain_num = 2
        #f.write("%d\t%d\t%0.6f\t%0.6f\t%0.6f\n"%(k,grain_num,g_B[0,i],g_B[1,i],g_B[2,i]))
        f.write("%d %d %0.10f %0.10f %0.10f\n"% (B[i,3],grain_num,B[i,0],B[i,1],B[i,2]))
        k += 1
    f.close()
    
    print("Done writing bicrystal "+name)
    return A,B


def min_shuffle_input_negative(Ai,Bi,Af,Bf,gb_loc,h,box,folder,sigma,inc,elem,disc_start,disc_end):
    """
    Reads min shuffle input for negative step heights

    Parameters
    ----------
    Ai : 2D array
        Initial configuration of Grain 1.
    Bi : 2D array
        Initial configuration of Grain 2.
    Af : 2D array
        Final configuration of Grain 1.
    Bf : 2D array
        Final configuration of Grain 2.
    gb_loc : float
        GB location.
    h : float
        Step height of disconnection mode.
    box : 2D array
        Box dimensions.
    folder : string
        output folder.
    sigma : int
        Gb sigma value.
    inc : float
        GB inclination.
    elem : string
        Element.
    disc_start : float
        Disconnection start location.
    disc_end : float
        Disconnection end location.

    Returns
    -------
    A : 2D array
        Grain 1.
    B : 2D array
        Grain 2.

    """
    data_A = []
    data_B = []
    eps = 0
    count = 0
    #print(gb_loc,gb_loc+h)
    for i in range(Bi.shape[0]):
        if  Bi[i,0]>gb_loc+h-eps and Bi[i,0]<gb_loc-eps and Bi[i,1]>=disc_start and Bi[i,1]<= disc_end:
            count += 1
            data_A.append([Bi[i,0],Bi[i,1],Bi[i,2],Bi[i,3]])
    A = np.array(data_A)
    for i in range(Af.shape[0]):
        for j in range(len(data_A)):
            if abs(Af[i,3]-A[j,3])<0.5:
                count += 1
                data_B.append([Af[i,0],Af[i,1],Af[i,2],Af[i,3]])
                break
            
    for i in range(Bf.shape[0]):
        for j in range(len(data_A)):
            if abs(Bf[i,3]-A[j,3])<0.5:
                count += 1
                data_B.append([Bf[i,0],Bf[i,1],Bf[i,2],Bf[i,3]])
                break
            
        
    
    B = np.array(data_B)
    prefix = "min_shuffle"
    file = "data." + elem + "s" +str(sigma) + "inc" + str(inc) + "_" +prefix# + "_"+ str(mode)
    name = folder + file
    #print(count)
    natoms = A.shape[0]+B.shape[0]
    #print("Check for same number of particles in the system")
    #print(A.shape[0],B.shape[0])
    f = open(name,"w")
    #f.write("# LAMMPS data file Sigma = %d, inclination = %f\n"%(sigma,inc))
    f.write("#LAMMPS data file\n")
    f.write("%d atoms\n"%(natoms))
    f.write("2 atom types\n")
    eps2 = 1
    xlo = gb_loc +h - eps2
    xhi = gb_loc - h + eps2
    mode = 2
    scale = 1.25
    f.write("%0.10f %0.10f xlo xhi\n"%(scale*xlo,scale*xhi))
    f.write("%0.10f %0.10f ylo yhi\n"%(box[1,0],box[1,1]))
    f.write("%0.10f %0.10f zlo zhi\n"%(box[2,0]+1e-2,box[2,1]))
    f.write("0.0 0.0 0.0 xy xz yz\n\n")
    f.write("Atoms # atomic\n\n")
    k= 1
    for i in range(A.shape[0]):
        grain_num = 1
        #f.write("%d\t%d\t%0.6f\t%0.6f\t%0.6f\n"% (k,grain_num,g_A[0,i],g_A[1,i],g_A[2,i]))
        f.write("%d %d %0.10f %0.10f %0.10f\n"% (A[i,3],grain_num,A[i,0],A[i,1],A[i,2]))
        k += 1
    for i in range(B.shape[0]):
        grain_num = 2
        #f.write("%d\t%d\t%0.6f\t%0.6f\t%0.6f\n"%(k,grain_num,g_B[0,i],g_B[1,i],g_B[2,i]))
        f.write("%d %d %0.10f %0.10f %0.10f\n"% (B[i,3],grain_num,B[i,0],B[i,1],B[i,2]))
        k += 1
    f.close()
    
    print("Done writing bicrystal "+name)
    return A,B

def neb_structure_minimized(g_A,g_B,Xs,Ys,Af,Bf,box,step_height,burgers_vector,folder,sigma,inc,elem,mode,y_size,disc_start,disc_end,version):
    """
    Write neb read atom positions for a bicrystal for a disconnection mode of positive step height

    Parameters
    ----------
    g_A : 2D array
        Initial configuration of Grain 1.
    g_B : 2D array
        Initial configuration of Grain 2.
    Xs : 2D array
        Initial atom positions in transformed region.
    Ys : 2D array
        Final atom positions in transformed region.
    Af : 2D array
        Final configuration of Grain 1.
    Bf : 2D array
        Final configuration of Grain 2.
    box : 2D array
        Box dimensions.
    step_height : float
        Step height of disconnection mode.
    burgers_vector : float
        Burgers vector of disconnection mode.
    folder : string
        output folder.
    sigma : int
        GB sigma value.
    inc : float
        GB inclination.
    elem : string
        Element.
    mode : string
        Disconnection mode (b<burgers_vector>h<step_height>).
    y_size : int
        Size of the system in term of period of GB.
    disc_start : float
        Dislocation start location.
    disc_end : float
        Dislocation end location.
    version : string 
        Name of step being created.

    Returns
    -------
    file : string
        name of output file.
    grainA : 2D array
        Grain 1
    grainb_ 2D array
        Grain 2

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
            #print(k)
            for i in range(g_B.shape[0]):
                grain_num = 2
                #f.write("%d\t%d\t%0.6f\t%0.6f\t%0.6f\n"%(k,grain_num,g_B[0,i],g_B[1,i],g_B[2,i]))
                f.write("%d %d %0.10f %0.10f %0.10f\n"% (g_B[i,3],grain_num,g_B[i,0],g_B[i,1],g_B[i,2]))
                grain_B.append([g_B[i,0],g_B[i,1],g_B[i,2],g_B[i,3]])
                k += 1
            #print(k)
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
            #print(k)
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
            #print(k)
            f.close()
            
            # Create the array
            
    print("Done writing bicrystal "+name)
    return file,np.array(grain_A),np.array(grain_B)


def neb_structure_minimized_negative(g_A,g_B,Xs,Ys,Af,Bf,box,step_height,burgers_vector,folder,sigma,inc,elem,mode,y_size,disc_start,disc_end,version,initial,final):
    """
    Write neb read atom positions for a bicrystal for a disconnection mode of negative step height

    Parameters
    ----------
   g_A : 2D array
       Initial configuration of Grain 1.
   g_B : 2D array
       Initial configuration of Grain 2.
   Xs : 2D array
       Initial atom positions in transformed region.
   Ys : 2D array
       Final atom positions in transformed region.
   Af : 2D array
       Final configuration of Grain 1.
   Bf : 2D array
       Final configuration of Grain 2.
   box : 2D array
       Box dimensions.
   step_height : float
       Step height of disconnection mode.
   burgers_vector : float
       Burgers vector of disconnection mode.
   folder : string
       output folder.
   sigma : int
       GB sigma value.
   inc : float
       GB inclination.
   elem : string
       Element.
   mode : string
       Disconnection mode (b<burgers_vector>h<step_height>).
   y_size : int
       Size of the system in term of period of GB.
   disc_start : float
       Dislocation start location.
   disc_end : float
       Dislocation end location.
   version : string 
       Name of step being created.
    initial : 2D array
        Initial atom positions.
    final : 2D array
        Final atom positions.

    Returns
    -------
    file : string
        name of output file.

    """
    suffix = ["_step0",version]
    #suffix = [version]
    for ii in range(len(suffix)):
        file = "data." + elem + "s" +str(sigma) + "inc" + str(inc) + "_" +suffix[ii]
        name = folder + file
        natoms = g_A.shape[0]+g_B.shape[0]
        k= 1
        #print(g_A.shape[0],g_B.shape[0])
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
            print("Done writing bicrystal "+name)
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
            #print(count,len(Xs),len(Ys))
            f.close()
            
            print("Done writing bicrystal "+name)
    
    return file

def write_neb_input_file(folder,file,image_num,outfile):
    """
    Create neb input file 
    
    Parameters
    ----------
    folder : string
        Location of directory where input file is to be stored.
    file : string
        Name of file.
    image_num : int
        Step number.
    outfile : string
        Name of  output file.

    Returns
    -------
    out_file : string
        Path to output file.

    """
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

def read_LAMMPS_datafile(path_r,mode=1):
    """
    Function to read the data file containing the data from a lammps data file
    Output form for each of the lines is : 

    Parameters
    ----------
    path_r : string
        Path to input file.
    mode : int,optional
        Parameter that helps regulate how input file it read. The defalut is 1

    Returns
    -------
    data : 2D array
        Array with each line as [ID,type,x,y,z].

    """
    i = 0
    q = -1
    j = q
    data = []
    k=0
    message = "Reading " + path_r
    #print(message)
    with open(path_r,'r') as file:
        flag1 = 0
        flag2 = 0
        flag3 = 0

        atoms = 0
        flag1 = 1
        box = np.zeros((3,2))
        pot_eng = 0
        atom_count = 0
        atom_pos_csym = []
        count = 0
        types = 0
        for line in file:
            fields = line.split(' ')
            #print(fields)
            i = i+1
            if fields[0].strip("\n")=="Velocities":
                break
            if(len(fields) == 2 and fields[1].strip('\n')=='atoms'):
                atoms = int(fields[0])
                flag2 = 0
                #print(atoms)
            if(len(fields)==3 and fields[1]=='atom'):
                types = int(fields[0])
            if(len(fields) == 4 and (fields[2]=="xlo" or fields[2]=="ylo" or fields[2]=="zlo")):
                box[count,0] = float(fields[0])
                box[count,1] = float(fields[1])
                count  = count + 1
                if count>2:
                    flag3 = 0
                #print(box)

            if mode == 2:
                if(len(fields)==8):
                    atom_count += 1
                    atom_pos_csym.append([float(fields[0]),float(fields[1]),float(fields[2]),float(fields[3]),float(fields[4])])
                    #print(atom_count,atoms)
                    if atom_count == int(atoms):
                        #print("yes")
                        a = np.asarray(atom_pos_csym)
                        data.append([atoms,types,box,a])
            elif mode == 1:
                if(len(fields)==5):
                    atom_count += 1
                    atom_pos_csym.append([float(fields[0]),float(fields[1]),float(fields[2]),float(fields[3]),float(fields[4])])
                    if atom_count == int(atoms):
                        a = np.asarray(atom_pos_csym)
                        data.append([atoms,types,box,a])
        #print(atom_count,atoms,len(atom_pos_csym))
    return data


def write_lammps_input(folder,g_A,g_B,elem,sigma,inc,axis,mode,box,prefix):
    """
    Function to print the given particle positions given 
    
    Generates a lammps data file   

    Parameters
    ----------
    folder : string
        Location of input file.
    g_A : 2D array
        Grain 1 atom locations.
    g_B : 2D array
        Grain 2 atom locations.
    elem : string
        Element.
    sigma : int
        GB sigma.
    inc : float
        GB inclination.
    axis : 1D Array
        Tilt axis.
    mode : int
        mode = 1 if cut along x axis and 2 if cut along y axis
    box : 2D array
        Box dimensions.
    prefix : string
        suffix to be added to the input file name.

    Returns
    -------
    file : string
        name of output file.
    grainA : 2D array
        Grain 1
    grainb_ 2D array
        Grain 2
    box : 2D array
        Box dimensions.

    """
    file = "data." + elem + "s" +str(sigma) + "inc" + str(inc) + "_" +prefix# + "_"+ str(mode)
    name = folder + file
    natoms = g_A.shape[0]+g_B.shape[0]
    f = open(name,"w")
    eps = 0.1
    #f.write("# LAMMPS data file Sigma = %d, inclination = %f\n"%(sigma,inc))
    f.write("#LAMMPS data file\n")
    f.write("%d atoms\n"%(natoms))
    f.write("2 atom types\n")
    f.write("%0.10f %0.10f xlo xhi\n"%(-box[mode-1,0]-eps,box[mode-1,0]+eps))
    f.write("%0.10f %0.10f ylo yhi\n"%(-box[mode-1,1],box[mode-1,1]))
    f.write("%0.10f %0.10f zlo zhi\n"%(-box[mode-1,2],box[mode-1,2]))
    box = np.array([[-box[mode-1,0]-eps,box[mode-1,0]-eps],[-box[mode-1,1],box[mode-1,1]],[-box[mode-1,2],box[mode-1,2]]])
    f.write("0.0 0.0 0.0 xy xz yz\n\n")
    f.write("Atoms # atomic\n\n")
    k= 1
    grain_A = []
    grain_B = []
    '''
    g_A = g_A[g_A[:,2].argsort()] # First sort doesn't need to be stable.
    g_A = g_A[g_A[:,1].argsort(kind='mergesort')]
    g_A = g_A[g_A[:,0].argsort(kind='mergesort')]
    g_B = g_B[g_B[:,2].argsort()] # First sort doesn't need to be stable.
    g_B = g_B[g_B[:,1].argsort(kind='mergesort')]
    g_B = g_B[g_B[:,0].argsort(kind='mergesort')]
    '''
    for i in range(g_A.shape[0]):
        grain_num = 1
        #f.write("%d\t%d\t%0.6f\t%0.6f\t%0.6f\n"% (k,grain_num,g_A[0,i],g_A[1,i],g_A[2,i]))
        f.write("%d %d %0.10f %0.10f %0.10f\n"% (k,grain_num,g_A[i,0],g_A[i,1],g_A[i,2]))
        grain_A.append([g_A[i,0],g_A[i,1],g_A[i,2],k])
        k += 1
    for i in range(g_B.shape[0]):
        grain_num = 2
        #f.write("%d\t%d\t%0.6f\t%0.6f\t%0.6f\n"%(k,grain_num,g_B[0,i],g_B[1,i],g_B[2,i]))
        f.write("%d %d %0.10f %0.10f %0.10f\n"% (k,grain_num,g_B[i,0],g_B[i,1],g_B[i,2]))
        grain_B.append([g_B[i,0],g_B[i,1],g_B[i,2],k])
        k += 1
    f.close()
    
    print("Done writing bicrystal " + name)
    return file,np.array(grain_A),np.array(grain_B),box

def write_lammps_input_ordered(folder,g_A,g_B,elem,sigma,inc,axis,mode,box,prefix,xpos,h,start,stop):
    """
    Function to print the given particle positions given 
    mode = 1 if cut along x axis and 2 if cut along y axis
    Generates a lammps data file     

    Parameters
    ----------
    folder : string
        Location of input file.
    g_A : 2D array
        Grain 1 atom locations.
    g_B : 2D array
        Grain 2 atom locations.
    elem : string
        Element.
    sigma : int
        GB sigma.
    inc : float
        GB inclination.
    axis : 1D Array
        Tilt axis.
    mode : int
        mode = 1 if cut along x axis and 2 if cut along y axis
    box : 2D array
        Box dimensions.
    prefix : string
        suffix to be added to the input file name.
    xpos : float
        Position of GB.
    h : float
        Step height.
    start : float
        Disconnection start location.
    stop : float
        Disconnection end location.

    Returns
    -------
    file : string
        name of output file.
    box : 2D array
        Box dimensions.

    """
    file = "data." + elem + "s" +str(sigma) + "inc" + str(inc) + "_" +prefix# + "_"+ str(mode)
    name = folder + file
    natoms = g_A.shape[0]+g_B.shape[0]
    f = open(name,"w")
    eps = 0.1
    #f.write("# LAMMPS data file Sigma = %d, inclination = %f\n"%(sigma,inc))
    f.write("#LAMMPS data file\n")
    f.write("%d atoms\n"%(natoms))
    f.write("2 atom types\n")
    f.write("%0.10f %0.10f xlo xhi\n"%(-box[mode-1,0]-eps,box[mode-1,0]+eps))
    f.write("%0.10f %0.10f ylo yhi\n"%(-box[mode-1,1],box[mode-1,1]))
    f.write("%0.10f %0.10f zlo zhi\n"%(-box[mode-1,2],box[mode-1,2]))
    box = np.array([[-box[mode-1,0]-eps,box[mode-1,0]-eps],[-box[mode-1,1],box[mode-1,1]],[-box[mode-1,2],box[mode-1,2]]])
    f.write("0.0 0.0 0.0 xy xz yz\n\n")
    f.write("Atoms # atomic\n\n")
    k= 1
    grain_A = []
    grain_B = []
    bicrystal = []
    for i in range(g_A.shape[0]):
        grain_num = 1
        row = np.array([g_A[i,0],g_A[i,1],g_A[i,2],g_A[i,3],grain_num])
        bicrystal.append(row)
    for i in range(g_B.shape[0]):
        if abs(g_B[i,0]-xpos-h)<0.01 and g_B[i,1]>start and g_B[i,1]<stop:
            grain_num = 1
        else:
            grain_num = 2
        row = np.array([g_B[i,0],g_B[i,1],g_B[i,2],g_B[i,3],grain_num])
        bicrystal.append(row)
    b = np.array(bicrystal)
    b = b[b[:,3].argsort()] 
    for i in range(b.shape[0]):
        f.write("%d %d %0.10f %0.10f %0.10f\n"% (b[i,3],b[i,4],b[i,0],b[i,1],b[i,2]))
    '''
    for i in range(g_A1.shape[0]):
        grain_num = 1
        #f.write("%d\t%d\t%0.6f\t%0.6f\t%0.6f\n"% (k,grain_num,g_A[0,i],g_A[1,i],g_A[2,i]))
        f.write("%d %d %0.10f %0.10f %0.10f\n"% (g_A1[i,3],grain_num,g_A1[i,0],g_A1[i,1],g_A1[i,2]))
    g_B1 = g_B[g_B[:,3].argsort()]
    for i in range(g_B1.shape[0]):
        grain_num = 2
        #f.write("%d\t%d\t%0.6f\t%0.6f\t%0.6f\n"%(k,grain_num,g_B[0,i],g_B[1,i],g_B[2,i]))
        f.write("%d %d %0.10f %0.10f %0.10f\n"% (g_B1[i,3],grain_num,g_B1[i,0],g_B1[i,1],g_B1[i,2]))
    '''
    f.close()
    
    print("Done writing bicrystal "+name)
    return file,box

def read_LAMMPS_dumpfile(path_r):
    """
    Reads the data file containing the data from a lammps dump file

    Parameters
    ----------
    path_r : string
        Location of LAMMPS dump file

    Returns
    -------
    data : 2D array
        Per Atom data with format [ID,type,x,y,z,centrosymmetry, energy]

    """
    i = 0
    q = -1
    j = q
    data = []
    k=0
    #message = "Reading " + path_r 
    #print(message)
    atom_count = 0
    with open(path_r,'r') as file:
        flag1 = 0
        flag2 = 0
        flag3 = 0
        t = 0
        atoms = 0
        for line in file:
            fields = line.split(' ')
            if(len(fields) == 2 and fields[1] == "TIMESTEP\n"):
                flag1 = 1
                #print(message)
            if(len(fields) == 1 and flag1==1):
                t = float(fields[0])
                flag1=0
                #message = "Timestep = " + str(t)
                #if t==0 or (t>145000 and t<150000):
                if t>-1:
                    i = i+1
                    #print(message)
                    box = np.zeros((3,2))
                    pot_eng = 0
                    atom_count = 0
                    atom_pos_csym = []
                else:
                    print("Unable to read lammps dump file")
                    continue
            if(len(fields) == 4 and fields[3] == "ATOMS\n"):
                flag2 = 1
            if(len(fields) == 1 and flag2==1):
                atoms = float(fields[0])
                flag2 = 0
            if(len(fields)==9 and fields[1]=="BOX"):
                flag3 = 1
                count = 0
            
            if(len(fields) == 3 and flag3==1):
                box[count,0] = float(fields[0])
                box[count,1] = float(fields[1])
                count  = count + 1
                if count>2:
                    flag3 = 0
        
            if(len(fields)==7):
                atom_count += 1
                atom_pos_csym.append([float(fields[0]),float(fields[1]),float(fields[2]),float(fields[3]),float(fields[4]),float(fields[5]),float(fields[6])])
                if atom_count == int(atoms):
                    a = np.asarray(atom_pos_csym)
                    data.append([t,atoms,box,a])  
                        
    return data
