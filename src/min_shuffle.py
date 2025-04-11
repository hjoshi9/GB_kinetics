import numpy as np
import numpy.linalg as la
import ot
import ot.plot
import matplotlib.pyplot as plt

from src.read_write_lammpsdatafiles import read_LAMMPS_datafile,read_LAMMPS_dumpfile



def find_gb_location(filepath):
    """
    Locates the position of GB from a lammps dumpfile

    Parameters
    ----------
    filepath : string
        Path to the LAMMPS data file under consideration.

    Returns
    -------
    average_gb_loc : float
        Average location of GB.
    extreme_gb_loc_hi : float
        Upper extent of GB.
    extreme_gb_loc_lo : float
        Lower extent of GB.

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
        if atoms[i,5]>2 and (box[0,0]+boundary_layer_thickness < atoms[i,2] < box[0,1]-boundary_layer_thickness):
            gb_atoms +=1
            average_gb_loc += atoms[i,2]
            if atoms[i,2]>extreme_gb_loc_hi:
                extreme_gb_loc_hi = atoms[i,2] 
            if atoms[i,2]<extreme_gb_loc_lo:
                extreme_gb_loc_lo = atoms[i,2] 
    average_gb_loc = average_gb_loc/gb_atoms
    
    return average_gb_loc,extreme_gb_loc_hi,extreme_gb_loc_lo

def replicate(rep_scheme,X,Y,box,dim=3):
    """
    
    Replicate the box provided in the system
    

    Parameters
    ----------
    rep_scheme : 1D array
        Multiplicative factors that determine which dimensions are to be extended to what extent [lo,xhi,ylo,yhi,xlo,zhi].
    X : 2D array
        Initial configuration atoms.
    Y : 2D array
        final configuration atoms.
    box : 2D array
        Dimension of box.
    dim : int
        Dimensionality of system (3D by default).

    Returns
    -------
    X_rep : 2D array
        Replicated Initial configuration atoms
    Y_rep : 2D array
        Replicated final configuration atoms.
    rep_box : 2D array
        Replicated box.

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

def pbcdist(dp,Lx,Ly,Lz,dim=3):
    """
    Finds the distance between particles according to periodic boundary conditions

    Parameters
    ----------
    dp : 1D array
        displcement
    Lx : float
        Box length in x
    Ly : float
        Box length in y
    Lz : float
        Box length in z
    dim : int
        Dimensionality of the simulation (default =3)

    Returns
    -------
    d : 1D array
        wrapped displacments

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

    Parameters
    ----------
    d : 2D array
        Array containing distances between initial and final configuration.
    box : 2D array
        Box dimensions.
    dim : int
        Dimensionality of the simulation (default =3)

    Returns
    -------
    dpbc : 2D array
        Array containing wrapped distances between initial and final configuration

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

    Parameters
    ----------
    g : 2D array
        Gamma value
    cutoff : float
        cutoff value

    Returns
    -------
    G : 2D array
        Thresholded Gamma value

    """
    G = abs(g)
    row_ind,col_ind = np.where(g<cutoff)
    for i in range(len(row_ind)):
        G[row_ind[i],col_ind[i]] = 0
    return G


def min_shuffle(dim,a,sig,misorient,folder,elem,reg_param,max_iters,box_expansion_factor):
    """
    Min shuffle algorithm implementation, make sure that the system has POT installed

    Parameters
    ----------
    dim : int
        Dimensionality of the simulation (default =3)
    a : float
        lattice parameter
    sig : int
        sigma value of GB
    misorient : float
        misorientation of GB
    folder : string
        Input file location 
    elem : string
        Element under consideration
    reg_param : float
        Regularization parameter for sinkhorm algorithm
    max_iters : int
        Maximum number of iterations for sinkhorn algorithm
    box_expansion_factor : int
        Factor that adds extra room to the original box (default = 0)

    Returns
    -------
    Xs : 2D array
        Initial configuration atoms
    Ys : 2D array
        Final configuration atoms
    Dvec : 2D array
        Displacement of atoms from initial to final configuration.
    indicies 2D aray: 
        Array describing the mapping chosen.
    gamma_mod : 2D array
        Output of skinhorm algorithm containing OT information

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
            #print(Xbasis.shape)
            if Xbasis.shape[0]!= Ybasis.shape[0]:
                print("The number of atoms do not match for the two grains. One-to-one mapping not possible")
                exit
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

