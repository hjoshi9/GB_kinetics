import numpy as np

def read_LAMMPS_dumpfile(path_r):
    """
        Reads atomic configurations from a LAMMPS dump file.

        Args:
            path_r (str): Path to the LAMMPS dump file.

        Returns:
            list: A list of snapshots, each containing:
                - timestep (float)
                - number of atoms (int)
                - box dimensions (3x2 np.ndarray)
                - atomic data (np.ndarray of shape [n_atoms, 7])
    """
    i = 0
    q = -1
    j = q
    data = []
    k = 0
    # message = "Reading " + path_r
    # print(message)
    atom_count = 0
    with open(path_r, 'r') as file:
        flag1 = 0
        flag2 = 0
        flag3 = 0
        t = 0
        atoms = 0
        for line in file:
            fields = line.split(' ')
            if (len(fields) == 2 and fields[1] == "TIMESTEP\n"):
                flag1 = 1
                # print(message)
            if (len(fields) == 1 and flag1 == 1):
                t = float(fields[0])
                flag1 = 0
                # message = "Timestep = " + str(t)
                # if t==0 or (t>145000 and t<150000):
                if t > -1:
                    i = i + 1
                    # print(message)
                    box = np.zeros((3, 2))
                    pot_eng = 0
                    atom_count = 0
                    atom_pos_csym = []
                else:
                    print("Unable to read lammps dump file")
                    continue
            if (len(fields) == 4 and fields[3] == "ATOMS\n"):
                flag2 = 1
            if (len(fields) == 1 and flag2 == 1):
                atoms = float(fields[0])
                flag2 = 0
            if (len(fields) == 9 and fields[1] == "BOX"):
                flag3 = 1
                count = 0

            if (len(fields) == 3 and flag3 == 1):
                box[count, 0] = float(fields[0])
                box[count, 1] = float(fields[1])
                count = count + 1
                if count > 2:
                    flag3 = 0

            if (len(fields) == 7):
                atom_count += 1
                atom_pos_csym.append(
                    [float(fields[0]), float(fields[1]), float(fields[2]), float(fields[3]), float(fields[4]),
                     float(fields[5]), float(fields[6])])
                if atom_count == int(atoms):
                    a = np.asarray(atom_pos_csym)
                    data.append([t, atoms, box, a])

    return data

def read_LAMMPS_datafile(path_r,mode=1):
    """
        Reads a LAMMPS data file and extracts atom positions and box dimensions.

        Args:
            path_r (str): Path to the data file.
            mode (int, optional): Parsing mode.
                - 1: Expecting 5 columns per atom line.
                - 2: Expecting 8 columns per atom line.

        Returns:
            list: A list containing:
                - number of atoms (int)
                - number of types (int)
                - box dimensions (3x2 np.ndarray)
                - atom positions (np.ndarray)
    """
    i = 0
    q = -1
    j = q
    data = []
    k = 0
    message = "Reading " + path_r
    # print(message)
    with open(path_r, 'r') as file:
        flag1 = 0
        flag2 = 0
        flag3 = 0

        atoms = 0
        flag1 = 1
        box = np.zeros((3, 2))
        pot_eng = 0
        atom_count = 0
        atom_pos_csym = []
        count = 0
        types = 0
        for line in file:
            fields = line.split(' ')
            # print(fields)
            i = i + 1
            if fields[0].strip("\n") == "Velocities":
                break
            if (len(fields) == 2 and fields[1].strip('\n') == 'atoms'):
                atoms = int(fields[0])
                flag2 = 0
                # print(atoms)
            if (len(fields) == 3 and fields[1] == 'atom'):
                types = int(fields[0])
            if (len(fields) == 4 and (fields[2] == "xlo" or fields[2] == "ylo" or fields[2] == "zlo")):
                box[count, 0] = float(fields[0])
                box[count, 1] = float(fields[1])
                count = count + 1
                if count > 2:
                    flag3 = 0
                # print(box)

            if mode == 2:
                if (len(fields) == 8):
                    atom_count += 1
                    atom_pos_csym.append(
                        [float(fields[0]), float(fields[1]), float(fields[2]), float(fields[3]), float(fields[4])])
                    # print(atom_count,atoms)
                    if atom_count == int(atoms):
                        # print("yes")
                        a = np.asarray(atom_pos_csym)
                        data.append([atoms, types, box, a])
            elif mode == 1:
                if (len(fields) == 5):
                    atom_count += 1
                    atom_pos_csym.append(
                        [float(fields[0]), float(fields[1]), float(fields[2]), float(fields[3]), float(fields[4])])
                    if atom_count == int(atoms):
                        a = np.asarray(atom_pos_csym)
                        data.append([atoms, types, box, a])
        # print(atom_count,atoms,len(atom_pos_csym))
    return data

def write_lammps_datafile(folder,file,suffix, g_A, g_B, input_box,mode = 2):
    """
        Writes a LAMMPS data file for a bicrystal configuration.

        Args:
            folder (str): Path to output folder.
            file (str): File name prefix.
            suffix (str): File suffix (e.g., 'min_shuffle').
            g_A (np.ndarray): Atom positions for grain A.
            g_B (np.ndarray): Atom positions for grain B.
            input_box (np.ndarray): Original box dimensions (3x3).
            mode (int, optional): Determines which box vector to expand for bicrystal. Default is 2 (z-direction).

        Returns:
            str: Path to the generated data file.
    """
    name = folder + file + "_"+suffix
    natoms = g_A.shape[0] + g_B.shape[0]
    f = open(name, "w")
    eps = 0.1
    # f.write("# LAMMPS data file Sigma = %d, inclination = %f\n"%(sigma,inc))
    f.write("#LAMMPS data file\n")
    f.write("%d atoms\n" % (natoms))
    f.write("2 atom types\n")
    if suffix == "min_shuffle":
        box = input_box
        box[0,:] *= 1.25
    else:
        box = np.array([[-input_box[mode - 1, 0] - eps, input_box[mode - 1, 0] - eps],
                        [-input_box[mode - 1, 1], input_box[mode - 1, 1]],
                        [-input_box[mode - 1, 2], input_box[mode - 1, 2]]])
    f.write("%0.10f %0.10f xlo xhi\n" % (box[0, 0], box[0, 1]))
    f.write("%0.10f %0.10f ylo yhi\n" % (box[1, 0], box[1, 1]))
    f.write("%0.10f %0.10f zlo zhi\n" % (box[2, 0], box[2, 1]))

    f.write("0.0 0.0 0.0 xy xz yz\n\n")
    f.write("Atoms # atomic\n\n")
    k = 1
    for i in range(g_A.shape[0]):
        grain_num = 1
        f.write("%d %d %0.10f %0.10f %0.10f\n" % (k, grain_num, g_A[i, 0], g_A[i, 1], g_A[i, 2]))
        k += 1
    for i in range(g_B.shape[0]):
        grain_num = 2
        f.write("%d %d %0.10f %0.10f %0.10f\n" % (k, grain_num, g_B[i, 0], g_B[i, 1], g_B[i, 2]))
        k += 1
    f.close()

    print("Done writing bicrystal " + name)
    return name

def find_gb_location(filepath):
    """
       Determines the location of a grain boundary based on centro-symmetry parameter.

       Args:
           filepath (str): Path to the LAMMPS dump file.

       Returns:
           tuple:
               - average_gb_loc (float): Average z-location of the grain boundary.
               - extreme_gb_loc_hi (float): Upper bound of GB atom positions.
               - extreme_gb_loc_lo (float): Lower bound of GB atom positions.
    """
    data = read_LAMMPS_dumpfile(filepath)
    box = data[-1][2]
    boundary_layer_thickness = 25
    atoms = data[-1][3]
    average_gb_loc = 0
    extreme_gb_loc_lo = 0
    extreme_gb_loc_hi = 0
    gb_atoms = 0
    for i in range(len(atoms)):
        if atoms[i, 5] > 2 and (
                box[0, 0] + boundary_layer_thickness < atoms[i, 2] < box[0, 1] - boundary_layer_thickness):
            gb_atoms += 1
            average_gb_loc += atoms[i, 2]
            if atoms[i, 2] > extreme_gb_loc_hi:
                extreme_gb_loc_hi = atoms[i, 2]
            if atoms[i, 2] < extreme_gb_loc_lo:
                extreme_gb_loc_lo = atoms[i, 2]
    average_gb_loc = average_gb_loc / gb_atoms

    return average_gb_loc, extreme_gb_loc_hi, extreme_gb_loc_lo

def read_neb_output_data(path_r,mode):
    """
        Reads output from a NEB (nudged elastic band) LAMMPS simulation.

        Args:
            path_r (str): Path to NEB output dump file.
            mode (int): Determines number of atom attributes to read (7 or 8 columns).

        Returns:
            list: A list of snapshots, each with:
                - timestep (float)
                - number of atoms (int)
                - box dimensions (3x2 np.ndarray)
                - atomic data (np.ndarray of shape [n_atoms, N])
    """
    i = 0
    q = -1
    j = q
    data = []
    k=0
    message = "Reading " + path_r
    #print(data)
    #print(message)
    with open(path_r,'r') as file:
        flag1 = 0
        flag2 = 0
        flag3 = 0
        t = 0
        atoms = 0
        for line in file:
            fields = line.split(' ')
            i = i+1
            #print(fields)
            if(len(fields) == 2 and fields[1] == "TIMESTEP\n"):
                flag1 = 1
                box = np.zeros((3,2))
                pot_eng = 0
                atom_count = 0
                atom_pos_csym = []
                #print(message)
            if(len(fields) == 1 and flag1==1):
                t = float(fields[0])
                flag1=0
                message = "Timestep = " + str(t)
                #print(t)

            if(len(fields) == 4 and fields[3] == "ATOMS\n"):
                flag2 = 1

            if(len(fields) == 1 and flag2==1):
                atoms = float(fields[0])
                flag2 = 0
                #print(atoms)

            if(len(fields)==9 and fields[8]=="pp\n"):
                flag3 = 1
                count = 0

            if(len(fields) == 3 and flag3==1):
                box[count,0] = float(fields[0])
                box[count,1] = float(fields[1])
                count  = count + 1
                if count>2:
                    flag3 = 0
                    #print(box)
            if mode == 1:
                count_elem = 8
            elif mode == 2:
                count_elem = 7
            if(len(fields)==count_elem):
                atom_count += 1
                #print(box)
                #atom_pos_csym.append([float(fields[0]),float(fields[1]),float(fields[2]),float(fields[3]),float(fields[4]),float(fields[5])])#,float(fields[6])])
                atom_pos_csym.append([float(fields[0]),float(fields[1]),float(fields[2]),float(fields[3]),float(fields[4]),float(fields[5]),float(fields[6])])
                if atom_count == int(atoms):
                    a = np.asarray(atom_pos_csym)
                    data.append([t,atoms,box,a])
    #print(data)
    return data

def write_LAMMPSoutput_tstep(natoms,Atoms,box,folder,file,tstep):
    """
       Writes a single timestep's atomic configuration to a LAMMPS trajectory file.

       Args:
           natoms (int): Total number of atoms.
           Atoms (np.ndarray): Array of atomic data (id, type, x, y, z, PotEng, CentroSym).
           box (np.ndarray): Box dimensions (3x2).
           folder (str): Output folder.
           file (str): File name.
           tstep (int): Timestep to write.

       Returns:
           str: Path to the output trajectory file.
    """
    name = folder + file
    if tstep==0:
        mode = "w"
    else:
        mode = "a"
    f = open(name,mode)
    #f.write("# LAMMPS data file Sigma = %d, inclination = %f\n"%(sigma,inc))
    f.write("ITEM: TIMESTEP\n")
    f.write("%d\n"%(tstep))
    f.write("ITEM: NUMBER OF ATOMS\n")
    f.write("%d\n"%(natoms))
    f.write("ITEM: BOX BOUNDS xy xz yz mm pp pp\n")
    f.write("%0.10f %0.10f 0.0\n"%(box[0,0],box[0,1]))
    f.write("%0.10f %0.10f 0.0\n"%(box[1,0],box[1,1]))
    f.write("%0.10f %0.10f 0.0\n"%(box[2,0],box[2,1]))
    f.write("ITEM: ATOMS id type x y z PotEng\n")# CentroSym\n")
    for i in range(Atoms.shape[0]):
        #print(Atoms[i,:])
        f.write("%d %d %0.10f %0.10f %0.10f %0.10f %0.10f\n"%(Atoms[i,0],Atoms[i,1],Atoms[i,2],Atoms[i,3],Atoms[i,4],Atoms[i,5],Atoms[i,6]))
        #f.write("%d %d %0.10f %0.10f %0.10f %0.10f\n"%(Atoms[i,0],Atoms[i,1],Atoms[i,2],Atoms[i,3],Atoms[i,4],Atoms[i,5]))

    f.close()

    #print("Done writing bicrystal")
    return file

def read_gridsearch_results(filepath):
    """
       Reads grid search results from a custom text file (displacements and GB energy).

       Args:
           filepath (str): Path to the results file.

       Returns:
           np.ndarray: Array with columns [dispy, dispz, GB_energy].
    """
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            fields = line.split(' ')
            row = []
            for i in range(len(fields)):
                if fields[i] == "dispy":
                    row.append(float(fields[i + 2]))
                elif fields[i] == "dispz":
                    row.append(float(fields[i + 2]))
                elif fields[i] == "GBene":
                    row.append(float(fields[i + 2]))
                    data.append(row)
    return np.array(data)

def write_gridsearch_results(filepath, data):
    """
        Writes displacements and GB energy to a tabulated results file.

        Args:
            filepath (str): Output file path.
            data (np.ndarray): Array of shape (N, 3) containing:
                - dispy (float)
                - dispz (float)
                - GB energy (float)
    """
    with open(filepath, 'w') as file:
        file.write("dispy\t\tdispz\t\tGB Energy\n")
        for i in range(len(data)):
            file.write("%f\t\t%f\t\t%f\n"%(data[i,0],data[i,1],data[i,2]))

