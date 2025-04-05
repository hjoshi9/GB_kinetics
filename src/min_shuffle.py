import numpy as np
import numpy.linalg as la
import ot
import ot.plot
import matplotlib.pyplot as plt


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

def replicate(rep_scheme,X,Y,box,dim):
    """
    Replicate the box provided in the system
    """
    # Define outputs
    rep_box = 0*box
    
    # Define multipliers
    xlo_mult = rep_scheme[0]
    xhi_mult = rep_scheme[1]
    ylo_mult = rep_scheme[2]
    yhi_mult = rep_scheme[3]
    zlo_mult = rep_scheme[4]
    zhi_mult = rep_scheme[5]
    
    x_range = range(xlo_mult,xhi_mult+1)
    y_range = range(ylo_mult,yhi_mult+1)
    z_range = range(zlo_mult,zhi_mult+1)
    nx = len(x_range)
    ny = len(y_range)
    nz = len(z_range)
    
    # Find lenght of box along each of the axes
    Lx = box[0,1]-box[0,0]
    Ly = box[1,1]-box[1,0]
    Lz = box[2,1]-box[2,0]
    
    nx = len(X)
    ny = len(Y)
    
    X_rep = np.zeros((X.shape[0]*nx*ny*nz,dim))
    Y_rep = np.zeros((Y.shape[0]*nx*ny*nz,dim))
    # Replicate 
    ind_count = 0
    for i in x_range:
        for j in y_range:
            for k in z_range:
                x_add = i*Lx
                y_add = j*Ly
                z_add = k*Lz
                
                x_new = X + np.array(x_add,y_add,z_add)
                y_new = Y + np.array(x_add,y_add,z_add)
                
                start = ind_count*nx
                end = (ind_count+1)*nx
                X_rep[start:end,:] = x_new
                Y_rep[start:end,:] = y_new
                ind_count += 1

    # Update box
    rep_box[0,0] = box[0,0] + xlo_mult*Lx
    rep_box[0,1] = box[0,1] + xhi_mult*Lx
    rep_box[1,0] = box[1,0] + ylo_mult*Ly
    rep_box[1,1] = box[1,1] + yhi_mult*Ly
    rep_box[2,0] = box[2,0] + zlo_mult*Lz
    rep_box[2,1] = box[2,1] + zhi_mult*Lz
    
    
    return X_rep,Y_rep,rep_box

def pbcdist(dp,Lx,Ly,Lz,dim):
    """
    Finds the distance between particles according to periodic boundary conditions
    """
    d = dp
    n = dim
    L = np.array([Lx,Ly,Lz])
    for i in range(n):
        Li = L[i]
        for j in range(dp.shape[0]):
            if dp[j,i]< -0.5*Li:
                d[j,i] += Li
            elif dp[j,i]>= 0.5*Li:
                d[j,i] -= Li
    return d

def pbcwrap(d,box,dim):
    """
    Wraps set of input coordinates according to pbc
    """
    dpbc  = d
    
    xlo = box[0,0]
    ylo = box[1,0]
    zlo = box[2,0]
    xhi = box[0,1]
    yhi = box[1,1]
    zhi = box[2,1]
    Lx = xhi-xlo
    Ly = yhi-ylo
    Lz = zhi-zlo
    
    L = np.array([Lx,Ly,Lz])
    
    for i in range(dim):
        di = d[:,i]
        Li = L[i]
        lo = box[i,0]
        hi = box[i,1]
        for j in range(len(d)):
            xj = di[j]
            if xj<lo:
                while xj<lo:
                    xj += Li
            if xj >= hi:
                while xj >= hi:
                    xj -= Li
            dpbc[j,i] = xj
    return dpbc

def threshold(g,cutoff):
    """
    Thresholds the value of Gamma obtained from OT algorithm upto a cutoff
    """
    G = abs(g)
    row_ind,col_ind = np.where(g<cutoff)
    for i in range(len(row_ind)):
        G[row_ind[i],col_ind[i]] = 0
    return G

def min_shuffle_input(Ai,Bi,Af,Bf,gb_loc,h,box,folder,sigma,inc,elem,disc_start,disc_end):
    """
    Reads input from lammps datafile which contains the information about the atoms on which minshuffle algorithm is to be acted on
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
    print("Check for same number of particles in the system")
    print(A.shape[0],B.shape[0])
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
    
    print("Done writing bicrystal")
    print(name)
    return A,B


def min_shuffle_input_negative(Ai,Bi,Af,Bf,gb_loc,h,box,folder,sigma,inc,elem,disc_start,disc_end):
    """
    Reads min shuffle input for negative step heights
    """
    data_A = []
    data_B = []
    eps = 0
    count = 0
    print(gb_loc,gb_loc+h)
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
    print(count)
    natoms = A.shape[0]+B.shape[0]
    print("Check for same number of particles in the system")
    print(A.shape[0],B.shape[0])
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
    
    print("Done writing bicrystal")
    print(name)
    return A,B

def min_shuffle(dim,a,sig,misorient,folder,elem,reg_param,max_iters,box_expansion_factor):
    """
    Min shuffle algorithm implementation, make sure that the system has POT installed
    """
    dim = 3
    lattice_parameter = a; #angstroms, for Al 
    r0 = lattice_parameter/np.sqrt(2); #nearest neighbor distance in perfect FCC crystal

    # Define the imput file location
    folder_name = folder #"/Users/hj-home/Desktop/Research/NEB/STGB_dataset/Al_axis_001optimal_beta/"#"Al_axis_001/"
    sigma = [sig]
    mis = [misorient]
    reg_p = [reg_param]#,0.05,0.1]
    for sigma_num in range(len(sigma)):
        for reg_num in range(len(reg_p)):
            file_name = "data."+elem+"s"+str(sigma[sigma_num])+"inc0.0_min_shuffle"
            
            file = folder_name + file_name
            mode = 2
            
            # Read input data
            data = read_LAMMPS_datafile(file, 1)
            
            # Find box size and atom positions for each of the types of particles 
            N = data[0][0]
            types = data[0][1]
            atoms = data[0][3]
            box = data[0][2] + box_expansion_factor*np.array([[0,0],[-5,5],[0,0]])
            xlo = box[0,0]
            ylo = box[1,0]
            zlo = box[2,0]
            xhi = box[0,1]
            yhi = box[1,1]
            zhi = box[2,1]
            Lx = xhi-xlo
            Ly = yhi-ylo
            Lz = zhi-zlo
            
            
            type_list = [x+1 for x in range(types)]
            # Find the particles of each of the grains: X is grain 1 Y is grain 2
            Xbasis = [] # type 1
            Ybasis = [] # type 2
            indicies = []
            for i in range(N):
                if atoms[i,1]==1.0:
                    Xbasis.append(atoms[i,2:5])
                    indicies.append(atoms[i,0])
                elif atoms[i,1]==2.0:
                    Ybasis.append(atoms[i,2:5])
            Xbasis = np.array(Xbasis)
            Ybasis = np.array(Ybasis)
            print(Xbasis.shape)
            if Xbasis.shape[0]!= Ybasis.shape[0]:
                print("The number of atoms do not match for the two grains. One-to-one mapping not possible")
            else:
                N = Xbasis.shape[0]
            
            
            # Find displacent vectors with pbcs
            pbcon = True 
            dist_mat = np.zeros((N,N))
            for i in range(N):
                dvec = Ybasis-Xbasis[i,:]
                dvec_pbc = pbcdist(dvec,Lx,Ly,Lz,dim)
                for j in range(N):
                    dist_mat[i,j] = la.norm(dvec_pbc[j,:])
            
            if pbcon == True:
                dist_mat = dist_mat**2
            
            # Optimal Transportation problem solution
            reg_param = reg_p[reg_num] # This is the regularization parameter, changing the value 
                              # means accessing different modes of transport.
                              # 0.005 ~ Min-shuffle
                              # Higher values give higher energy permutations, related to kbT
            p = np.zeros(dim) # Recomputed, set to [0,0,0] if unknown and results will match TDP
            
            # Sinkhorn's algorithm
            a = np.ones(N)/N
            b = np.ones(N)/N
            iter_max = max_iters
            Gamma = ot.bregman.sinkhorn_log(a,b,dist_mat,reg_param,iter_max)
            
            # Recover the displacement vectors from OT matrix(gamma)
            cutoff = 1/(N*2)
            gamma_mod = threshold(Gamma,cutoff)
            I,J = np.where(gamma_mod!=0)
            #I,J = np.where(Gamma!=0)
            maxNp = np.max(gamma_mod)
            disp_vecs = np.zeros((len(I),11))
            for i in range(len(I)):
                K = gamma_mod[I[i],J[i]] # path prob
                Xcoords = np.zeros((1,dim))
                Ycoords = np.zeros((1,dim))
                Xcoords[0,:] = np.array([Xbasis[I[i],0],Xbasis[I[i],1],Xbasis[I[i],2]])
                Ycoords[0,:] = np.array([Ybasis[J[i],0],Ybasis[J[i],1],Ybasis[J[i],2]])
                index = indicies[I[i]]
                
                # Vector connecting X and Y
                disp_vec_new = Ycoords-Xcoords
                if pbcon==True:
                    disp_pbc = pbcdist(disp_vec_new, Lx, Ly, Lz, dim)
                newYcoords = Xcoords + disp_pbc
                
                disp_vecs[i,:] = np.array([disp_pbc[0,0],disp_pbc[0,1],disp_pbc[0,2],K,Xcoords[0,0],
                                           Xcoords[0,1],Xcoords[0,2],newYcoords[0,0],newYcoords[0,1],newYcoords[0,2],index])
            
            # Data in different frames
            ndisps = disp_vecs.shape[0] 
            Kvec = disp_vecs[:,3]
            Kvecrenorm = la.norm(Kvec)
            
            Xcoords = disp_vecs[:,4:7]
            Ycoords = disp_vecs[:,7:10] 
            dTDP = disp_vecs[:,0:3]
            XTDP = Xcoords
            YTDP = Ycoords 
            dCDP = dTDP - p
            XCDP = Xcoords
            YCDP = Ycoords-p
            prob = np.zeros((len(Kvec),3))
            idx = disp_vecs[:,10]
            for i in range(len(Kvec)):
                prob[i,:] = Kvec[i]*dTDP[i,:]
            Dvec = np.sum(prob,0) #probabilistic expression for total net displacement/atom
            dSDP = dTDP-Dvec
            XSDP = Xcoords
            YSDP = Ycoords-Dvec
            
            Mvecest = Dvec-p
            microvecs = np.array([p[0],p[1],p[2],
                                  Mvecest[0],Mvecest[1],Mvecest[2],
                                  Dvec[0],Dvec[1],Dvec[2]])
            Xp = XTDP
            Yp = YTDP
            
            
            # Plotting routine
            #%matplotlib qt
            #fig = plt.figure(dpi = 300)
            #ax = fig.add_subplot(projection='3d')
            elev = 90
            azim = 00
            roll = 00
            #ax.view_init(elev,azim,roll)
            
            Xs = np.zeros((ndisps,4))
            Ys = np.zeros((ndisps,4)) #copies of coordinates to overwrite with sheared coordinates
            Xss = []
            Yss = []
            indicies_s = []
            k = 0
            prob_cutoff = 1e-8
            for i in range(ndisps):
                path_prob = Kvec[i]
                if path_prob > prob_cutoff:
                    a = np.zeros((1,3))
                    b = np.zeros((1,3))
                    a[0,:] = Xp[i,:]
                    b[0,:] = Yp[i,:]
                    #print(np.round(b-a,4))
                    if pbcon ==  True:
                        dp_new = pbcdist(b-a, Lx, Ly, Lz, dim)
                    b_new = a + dp_new 
                    #print(a,Xs)
                    Xs[k,:3] = a
                    Xs[k,3] = idx[i]
                    Ys[k,:3] = b_new
                    Ys[k,3] = idx[i]
                    #print(b_new)
                    flag = 0
                    # Check for PBCs along y and z
                    if b_new[0,1]>yhi:
                        b_new[0,1] -= Ly
                        flag = 1
                    if b_new[0,2]>zhi:
                        b_new[0,2] -= Lz
                        flag = 1
                    if b_new[0,1]<ylo:
                        b_new[0,1] += Ly
                        flag = 1
                    if b_new[0,2]<=zlo+0.1:
                        b_new[0,2] += Lz
                        flag = 1
                    #if b_new
                    Ys[k,:3] = b_new
                    #print(b_new)
                    k += 1
                    
                    x,y,z = [a[0,0],b_new[0,0]],[a[0,1],b_new[0,1]],[a[0,2],b_new[0,2]]
                    #print(x,y,z)
                    '''
                    if flag==0:
                        ax.plot(x,y,z,'green',linewidth = 0.75)
                    else:
                        ax.plot(x,y,z,'green',linewidth = 0.25)
                    '''
            #for i in range(len(Xs)):
            
            # Remove 
            '''
            # Plot simulation box
            ax.plot([xlo,xhi],[yhi,yhi],[zhi,zhi],'k',linewidth=1)
            ax.plot([xlo,xhi],[ylo,ylo],[zhi,zhi],'k',linewidth=1)
            ax.plot([xlo,xhi],[yhi,yhi],[zlo,zlo],'k',linewidth=1)
            ax.plot([xlo,xhi],[ylo,ylo],[zlo,zlo],'k',linewidth=1)
            ax.plot([xlo,xlo],[ylo,yhi],[zhi,zhi],'k',linewidth=1)
            ax.plot([xlo,xlo],[ylo,yhi],[zlo,zlo],'k',linewidth=1)
            ax.plot([xhi,xhi],[ylo,yhi],[zlo,zlo],'k',linewidth=1)
            ax.plot([xhi,xhi],[ylo,yhi],[zhi,zhi],'k',linewidth=1)
            ax.plot([xlo,xlo],[ylo,ylo],[zlo,zhi],'k',linewidth=1)
            ax.plot([xlo,xlo],[yhi,yhi],[zlo,zhi],'k',linewidth=1)
            ax.plot([xhi,xhi],[ylo,ylo],[zlo,zhi],'k',linewidth=1)
            ax.plot([xhi,xhi],[yhi,yhi],[zlo,zhi],'k',linewidth=1)
            '''
            # Plot particles
            '''
            ax.scatter(Xs[:,0],Xs[:,1],Xs[:,2],color='red',s = 2,label="Initial")
            ax.scatter(Ys[:,0],Ys[:,1],Ys[:,2],color='blue',s = 2,label="Final")
            ax.set_xlabel("X") 
            ax.set_ylabel("Y") 
            ax.set_zlabel("Z")
            ax.legend()
            n = "Al sigma" + str(sigma[sigma_num]) + "min shuffle patterns"
            plt.title(n)
            name = folder_name + "Als"+str(sigma[sigma_num])+"inc0.0_optimalBeta_min_shuffle_eps"+str(reg_param)+".png"
            print("NEB calculation done!")
            print(Xs.shape)
            print(Ys.shape)
            plt.figure(dpi = 200)
            ot.plot.plot2D_samples_mat(Xbasis, Ybasis, Gamma, c=[.5, .5, 1])
            plt.scatter(Xbasis[:, 0], Xbasis[:, 1], color='b', label='Source samples')
            plt.scatter(Ybasis[:, 0], Ybasis[:, 1], color='r', label='Target samples')
            plt.legend(loc='best')
            n = "Al sigma" + str(sigma[sigma_num]) + "min shuffle optimal beta eps="+str(reg_param)
            plt.title(n)
            plt.savefig(name)
            print(sigma[sigma_num],Dvec)
            '''
    return Xs,Ys,Dvec,indicies,gamma_mod

