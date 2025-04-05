import numpy as np

def read_LAMMPS_datafile(path_r,mode):
    """
    Function to read the data file containing the data from a lammps data file
    Output form for each of the lines is : [ID,type,x,y,z]
    """
    i = 0
    q = -1
    j = q
    data = []
    k=0
    message = "Reading " + path_r
    print(message)
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
    mode = 1 if cut along x axis and 2 if cut along y axis
    Generates a lammps data file     
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
    
    print("Done writing bicrystal")
    return file,np.array(grain_A),np.array(grain_B),box

def write_lammps_input_ordered(folder,g_A,g_B,elem,sigma,inc,axis,mode,box,prefix,xpos,h,start,stop):
    """
    Function to print the given particle positions given 
    mode = 1 if cut along x axis and 2 if cut along y axis
    Generates a lammps data file     
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
    
    print("Done writing bicrystal")
    return file,box
