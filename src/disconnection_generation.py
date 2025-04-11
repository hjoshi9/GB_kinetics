import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from src.solid_angle_calculations import *


def check_unique(point,grain,cutoff):
    """
    Checks if the atom position is unique or not

    Parameters
    ----------
    point : 1D array
        Position of an atom [x,y,z].
    grain : 2D array
        Atom positions already in a grain.
    cutoff : float
        parameter to decide if atoms are too close.

    Returns
    -------
    unique : int
        1 if point is new, 0 otherwise.

    """
    unique = 1
    for i in range(len(grain)):
        if abs(grain[i][0] - point[0])<cutoff and abs(grain[i][1] - point[1])<cutoff and abs(grain[i][2] - point[2])<cutoff: 
            unique = 0
            break
    
    return unique


def diagnostic_grain_writing(grain1,grain2,filename):
    """
    
    Write grain 1 and grain 2 atoms for diagnostic purposes
    
    Parameters
    ----------
    grain1 : 2D array
        Grain 1 atom positions.
    grain2 : 2D array
        Grain 2 atom positions.
    filename : string
        Output file.
        
    Returns
    -------
    1

    """
    f = open(filename,"w")
    f.write("%d\n\n"%(len(grain1)+len(grain2)))
    for i in range(len(grain1)):
        f.write("%f %f %f %d\n"%(grain1[i,0],grain1[i,1],grain1[i,2],1))
    for i in range(len(grain2)):
        f.write("%f %f %f %d\n"%(grain2[i,0],grain2[i,1],grain2[i,2],2))
    f.close()
    return 1
    

def diagnostic_plotting(grain1,grain2,minx,maxx,miny,maxy):
    """
    Plot the grain 1 and grain 2 atoms for diagnostics

    Parameters
    ----------
    grain1 : 2D array
        Grain 1 atom positions.
    grain2 : 2D array
        Grain 2 atom positions.
    minx : float
        Plot limit xlo.
    maxx : float
        Plot limit xhi.
    miny : float
        Plot limit ylo.
    maxy : float
        Plot limit yhi.

    Returns
    -------
    1

    """
    g = np.array(grain1)
    g2 = np.array(grain2)
    plt.figure(dpi=200,figsize=(3,3))
    plt.scatter(g[:,1],g[:,0],s=1,color="red")
    plt.scatter(g2[:,1],g2[:,0],s = 1,color="blue")
    plt.xlim([minx,maxx])
    plt.ylim([miny,maxy])
    
    return 1


def create_bicrystal(gb_data,axis,lat_par,lat_Vec):
    """
    Create a reference bicrystal for the gb in question

    Parameters
    ----------
    gb_data : 1D array
        vector coantaining gb data like sigma, misorientation, inclination, period.
    axis : 1D array
        Tilt axis.
    lat_par : float
        lattice parameter.
    lat_Vec : 2D array
        Primitive vectors for crystal system.

    Returns
    -------
    A : 2D array
        Grain 1 lattice vectors.
    B : 2D array
        Grain 2 lattice vectors.
    a : 2D array
        Rotated grain 1 lattice vectors.
    b : 2D array
        Rotated grain 2 lattice vectors.

    """
    dim = 3
    # fRotate basis vectors to get A and B
    axis_true = axis
    axis = axis/la.norm(axis)
    K=np.array([[0 ,-axis[2], axis[1]],
                [axis[2], 0 ,-axis[0]],
                [-axis[1], axis[0], 0]])
    angle = gb_data[1]
    theta1=1*(angle/2)*np.pi/180
    theta2 =1*(angle/2)*np.pi/180
    inc = gb_data[2]
    phi = (inc)*np.pi/180
    ra=np.eye(3) + np.sin(theta1+phi)*K + (1-np.cos(theta1+phi))*np.matmul(K,K)
    rb=np.eye(3) + np.sin(-theta2+phi)*K + (1-np.cos(-theta2+phi))*np.matmul(K,K)
    a = np.matmul(ra,lat_Vec)
    b = np.matmul(rb,lat_Vec)
    #burgers_vec = np.array([gb_data[4],gb_data[5],gb_data[6]])
    # Rotate crystals such that tilt axis lies along [001]
    z = np.array([0,0,1])
    ax = np.cross(axis,z)
    if axis[0] == z[0] and axis[1] == z[1] and axis[2] == z[2] :
        Rot_fin = np.eye(dim)
    else:
        ax = ax/la.norm(ax)
        V = np.array([[0 ,-ax[2], ax[1]],
                    [ax[2], 0 ,-ax[0]],
                    [-ax[1], ax[0], 0]])
        rot_angle = -np.arccos(np.dot(axis,z)/la.norm(axis))
        Rot = np.eye(dim) + np.sin(rot_angle)*V + np.matmul(V,V)*(1-np.cos(rot_angle))
        
        if axis_true[0] == 1 and axis_true[1] == 1 and axis_true[2] == 1 :
            Rot_gb = np.array([[np.cos(np.pi/4),-np.sin(np.pi/4),0],
                               [np.sin(np.pi/4),np.cos(np.pi/4),0],
                               [0,0,1]]) 
        if axis_true[0] == 1 and axis_true[1] == 1 and axis_true[2] == 0 :
            Rot_gb = np.array([[np.cos(np.pi/4),-np.sin(np.pi/4),0],
                               [np.sin(np.pi/4),np.cos(np.pi/4),0],
                               [0,0,1]]) 
        '''
        else:
            
            GB_normal = Rot.T@burgers_vec
            direction_along_GB = np.cross(GB_normal,z)
            rot_ax = z
            intended_gb_direction = np.array([0,1,0])
            U = np.array([[0 ,-rot_ax[2], rot_ax[1]],
                        [rot_ax[2], 0 ,-rot_ax[0]],
                        [-rot_ax[1], rot_ax[0], 0]])
            rot_angle = np.arccos(np.dot(direction_along_GB,intended_gb_direction)/la.norm(direction_along_GB))
            Rot_gb = np.eye(dim) + np.sin(rot_angle)*U + np.matmul(U,U)*(1-np.cos(rot_angle))\
            '''
        #Rot_gb = np.eye(dim)
        Rot_fin = np.matmul(Rot,Rot_gb)
        aa = Rot_gb
    A = np.matmul(Rot_fin.T,a)
    B = np.matmul(Rot_fin.T,b)
    return A,B,a,b

def generate_disconnection_ordered_initial(c1,c2,nCells,period,axis,lat_par,non_per_cutoff,bur,size,nodes,nImages,h,x_pos):
    """
    Create the initial as cut gb using data from main

    Parameters
    ----------
    c1 : 2D array
        Lattice vectors for grain1.
    c2 : 2D array
        Lattice vectors for grain2.
    nCells : int
        Number of cells created.
    period : float
        Period of the GB.
    axis : 1D array
        Tilt axis.
    lat_par : float
        lattice parameter.
    non_per_cutoff : float
        Length of box along the nonperiodic direction.
    bur : float
        burgers vector or disconnection.
    size : int
        Factor that controls box size along the GB. Box length along GB = 2*CSL_period*size
    nodes : 2D array
        End points of dislocations inserted in the GB.
    nImages : int
        Periodic  images of dislocations to be used.
    h : float
        Step height of disconnection mode.
    x_pos : float
        Position of gb along non periodic direction.

    Returns
    -------
    gA1 : 2D array
        Atom positions for grain 1.
    gB1 : 2D array
        Atom positions for grain 2.
    box : 2D array
        Box dimensions.

    """
    
    grainA1 = []
    grainB1 = []
    atom_num_1 = 0
    atom_num_2 = 0
    zfactor = 1
    box = np.array([[size*period,non_per_cutoff,la.norm(axis)*lat_par*zfactor],
                    [non_per_cutoff,size*period,la.norm(axis)*lat_par*zfactor]])
    eps = 0.1
    box_shift = 0
    #Creation
    for nx in range(-nCells,nCells):
        for ny in range(-nCells,nCells):
            for nz in range(-nCells,nCells):
                loc = np.array([[nx],[ny],[nz]])
                a = lat_par*np.matmul(c1,loc)[:,0]
                b = lat_par*np.matmul(c2,loc)[:,0]
                if a[0]>-box[1,0]+eps and a[0]<box[1,0]+eps and (a[2]) < box[1,2]+eps and (a[2]) > -box[1,2]+eps and (a[1]-box_shift)<box[1,1]+eps and (a[1]-box_shift)>-box[1,1]+eps:
                    disp = 0
                    grainA1.append([a[0],a[1]-disp,a[2]])
                    atom_num_2 +=1
                if b[0]<box[1,0]+eps and b[0]>-box[1,0]-eps and b[2]< box[1,2]+eps and (b[2]) > -box[1,2]+eps and (b[1]-box_shift)<box[1,1]+eps and (b[1]-box_shift)>-box[1,1]+eps:
                    disp = 0
                    grainB1.append([b[0],b[1]-disp,b[2]])
                    atom_num_2 +=1
    #Deletion
    gA = []
    gB = []
    #x_pos = 0#-lat_par*period/2
    f = 1
    y_eps = 0.
    y_eps2 = 0.4
    eps = 0.1
    start = nodes[0,0]
    stop = nodes[1,0]
    atom_count = 1
    for i in range(len(grainA1)):
        if grainA1[i][0]>=x_pos-eps:
            point = [grainA1[i][0],grainA1[i][1],grainA1[i][2]]
            flag = check_unique(point, gA, 0.5)
            if flag==1:
                point.append(atom_count)
                gA.append(point)
                atom_count += 1
        
    
    for i in range(len(grainB1)):
        if  grainB1[i][0]<x_pos-eps:
            point = [grainB1[i][0],grainB1[i][1],grainB1[i][2]]
            flag = check_unique(point, gB, 0.5)
            if flag==1:
                point.append(atom_count)
                gB.append(point)
                atom_count += 1
    #print(len(gA),len(gB),len(gA)+len(gB))
    gA1 = np.array(gA)
    gB1 = np.array(gB)
    #print(bur,h)
    return gA1,gB1,box


def generate_disconnection_ordered_other(c1,c2,nCells,period,axis,lat_par,non_per_cutoff,bur,size,nodes,nImages,h,gA_1,gB_1,x_pos,image_number,dipole_number):
    """
    Generate images with disconnection step

    Parameters
    ----------
    c1 : 2D array
        Lattice vectors for grain1.
    c2 : 2D array
        Lattice vectors for grain2.
    nCells : int
        Number of cells created.
    period : float
        Period of the GB.
    axis : 1D array
        Tilt axis.
    lat_par : float
        lattice parameter.
    non_per_cutoff : float
        Length of box along the nonperiodic direction.
    bur : float
        burgers vector or disconnection.
    size : int
        Factor that controls box size along the GB. Box length along GB = 2*CSL_period*size
    nodes : 2D array
        End points of dislocations inserted in the GB.
    nImages : int
        Periodic  images of dislocations to be used.
    h : float
        Step height of disconnection mode.
    x_pos : float
        Position of gb along non periodic direction.
    gA1 : 2D array
        Atom positions for grain 1.
    gB1 : 2D array
        Atom positions for grain 2.
    x_pos : float
        Position of gb along non periodic direction.
    image_number : int 
        Image number .
    dipole_number : int 
        Number of dislocation dipoles inserted.

    Returns
    -------
    gA1 : 2D array
        Atom positions for grain 1.
    gB1 : 2D array
        Atom positions for grain 2.
    box : 2D array
        Box dimensions.

    """
    grainA1 = []
    grainB1 = []
    atom_num_1 = 0
    atom_num_2 = 0
    zfactor = 1
    box = np.array([[size*period,non_per_cutoff,la.norm(axis)*lat_par*zfactor],
                    [non_per_cutoff,size*period,la.norm(axis)*lat_par*zfactor]])
    #x_pos = 0#-lat_par*period/2
    eps = 0# for m=1, n= 0: 0.1
    eps2 = 0.1
    box_shift = 0
    start = nodes[0,0]
    stop = nodes[1,0]
    diag_plt = False #True
    nodes_for_solid_angle = np.zeros((2,2))
    # for m = 2, n = -1
    if image_number < 2*size:
        fac = 10
    else:
        fac = 0
    if image_number < 2*size:
        nodes_modified = [nodes] #+ np.array([[-period/8,0],[period/8,0]])
        if dipole_number > 1:
            transition_node_start = np.array([[nodes[0,0],x_pos],[nodes[0,0],nodes[0,1]]])
            transition_node_stop = np.array([[nodes[1,0],nodes[1,1]],[nodes[1,0],x_pos]])
            nodes_modified.append(transition_node_start)
            nodes_modified.append(transition_node_stop)
    else: 
        nodes_modified = [nodes + np.array([[-period/4,0],[period/4,0]])]
        dipole_number = 1
    # if m = 3 or m = 2
    fac = 0
    # if m = -1 n = 1
    if image_number < 2*size:
        fac1 = 0
    else:
        fac1 = 0
    #Creation
    grainA1 = []
    grainB1 = []
    for nx in range(-nCells,nCells):
        for ny in range(-nCells,nCells):
            for nz in range(-nCells,nCells):
                loc = np.array([[nx],[ny],[nz]])
                a = lat_par*np.matmul(c1,loc)[:,0]
                b = lat_par*np.matmul(c2,loc)[:,0]
                if h<0:
                    # For m = 1
                    #if a[0]>x_pos+h-1*eps2 and a[0]<x_pos-eps2 and (a[2]) < box[1,2]+0.25*eps2 and (a[2]) > -box[1,2]+0*eps2 and (a[1]-box_shift)>start and (a[1]-box_shift)<stop:
                    #For m = 3
                    if a[0]>x_pos+h-eps2*1 and a[0]<x_pos-eps2 and (a[2]) < box[1,2]+0.25*eps2 and (a[2]) > -box[1,2]+0*eps2 and (a[1]-box_shift)>start-eps2*0 and (a[1]-box_shift)<stop-eps2*00:
                        disp = 0
                        point = np.array([a[1],a[0]])
                        for dp in range(dipole_number):
                            nodes_for_solid_angle = nodes_modified[dp]
                            solidAngle,disp_temp=solidangle_displacement(nImages, nodes_for_solid_angle, period, point, bur)
                            disp += disp_temp
                        if a[1]-disp>box[1,1]:
                            disp += 2*box[1,1]
                        elif a[1]-disp<-box[1,1]:
                            disp -= 2*box[1,1]
                        grainA1.append([a[0],a[1]-disp,a[2]])
                        atom_num_2 +=1
                else:
                    # works
                    eps2 = 0.1                    
                    #if b[0]<box[1,0]+eps and b[0]>-box[1,0]-eps and b[2]< box[1,2]+eps and (b[2]) > -box[1,2]+eps and (b[1]-box_shift)<box[1,1]+eps and (b[1]-box_shift)>-box[1,1]+eps:
                    if b[0]<x_pos+h-eps2 and b[0]>x_pos-eps2 and b[2]< box[1,2]+0.25*eps2 and (b[2]) > -box[1,2] and (b[1])>start-0.0 and (b[1])<stop:
                    # Trial
                    #if b[0]<x_pos+h+eps2 and b[0]>x_pos-eps2 and b[2]< box[1,2]+eps2 and (b[2]) > -box[1,2]+eps2 and (b[1]-box_shift)>start-0*eps2 and (b[1]-box_shift)<stop-fac*eps2:
                        disp = 0
                        point = np.array([b[1],b[0]])
                        for dp in range(dipole_number):
                            nodes_for_solid_angle = nodes_modified[dp]
                            solidAngle,disp_temp=solidangle_displacement(nImages, nodes_for_solid_angle, period, point, bur)
                            disp += 0*disp_temp
                        if b[1]-disp>box[1,1]:
                            disp += 2*box[1,1]
                        elif b[1]-disp<-box[1,1]:
                            disp -= 2*box[1,1]
                        point = [b[0],b[1]-disp,b[2]]
                        if check_unique(point, grainB1, 0.1) == 1:
                            
                            grainB1.append(point)
                        atom_num_2 +=1
    #print(len(grainB1))
    diag_plt = False
    if diag_plt == True:
        if len(grainB1)>0:
            diagnostic_plotting(grainB1, nodes, -15, 15,-20,20)
        elif len(grainA1)>0:
            diagnostic_plotting(grainA1, nodes, -15,15,-20,20)
    #Deletion
    gA = []
    gB = []
    y_eps = 0.
    eps = 0.1# for m=1, n= 0: 0.1
    start = nodes[0,0]
    stop = nodes[1,0]
    grainA_old = []
    grainB_old = []
    #print(start,stop,h,bur)
    
    if h>= 0 :
        for i in range(len(gA_1)):
            if gA_1[i,0]>=x_pos+h-0.1 or (((gA_1[i,1])<start or (gA_1[i,1])>stop) and gA_1[i,0]>x_pos-0.1):
                disp = 0
                point = np.array([gA_1[i,1],gA_1[i,0]])
                for dp in range(dipole_number):
                    nodes_for_solid_angle = nodes_modified[dp]
                    solidAngle,disp_temp=solidangle_displacement(nImages, nodes_for_solid_angle, period, point, bur)
                    disp += disp_temp
                #if abs(gA_1[i,0] - (x_pos+h-0.1))>0.5:
                if abs(gA_1[i,0]-x_pos-h)<0.2:
                    disp = 1*disp
                if gA_1[i,1]-disp>box[1,1]:
                    disp += 2*box[1,1]
                elif gA_1[i,1]-disp<-box[1,1]:
                    disp -= 2*box[1,1]
                    
                row = np.array([gA_1[i,0],gA_1[i,1]-disp,gA_1[i,2],gA_1[i,3]])
                gA.append(row)
            else:
                disp = 0
                point = np.array([gA_1[i,1],gA_1[i,0]])
                for dp in range(dipole_number):
                    nodes_for_solid_angle = nodes_modified[dp]
                    solidAngle,disp_temp=solidangle_displacement(nImages, nodes_for_solid_angle, period, point, bur)
                    disp += disp_temp
                if abs(gA_1[i,0]-x_pos)<0.2:
                    disp = 1*disp
                if gA_1[i,1]-disp>box[1,1]:
                    disp += 2*box[1,1]
                elif gA_1[i,1]-disp<-box[1,1]:
                    disp -= 2*box[1,1]
                grainA_old.append([gA_1[i,0],gA_1[i,1]-disp,gA_1[i,2],gA_1[i,3]])
        #print(len(grainA_old),len(grainB1)) 
        #diagnostic_grain_writing(np.array(grainA_old),np.array(grainB1),"grain_diagnostics.txt")
        diagnostic_plotting(grainA_old,grainB1, -35,35,-20,20)
        if diag_plt == True:
            diagnostic_plotting(gA,grainA_old, -15, 15,-20,20)
            
        for i in range(len(gB_1)):
            if gB_1[i,0]<x_pos-eps: #or (((gB_1[i,1])>start and gB_1[i,1]<=stop-y_eps) and gB_1[i,0]<x_pos+h-eps):
                disp = 0
                point = np.array([gB_1[i,1],gB_1[i,0]])
                for dp in range(dipole_number):
                    nodes_for_solid_angle = nodes_modified[dp]
                    solidAngle,disp_temp=solidangle_displacement(nImages, nodes_for_solid_angle, period, point, bur)
                    disp += disp_temp
                if gB_1[i,1]-disp>box[1,1]:
                    disp += 2*box[1,1]
                elif gB_1[i,1]-disp<-box[1,1]:
                    disp -= 2*box[1,1]
                row = np.array([gB_1[i,0],gB_1[i,1]-disp,gB_1[i,2],gB_1[i,3]])
                gB.append(row)
        
        # Add atoms to grain B due to transformation
        countb = 1
        for i in range(len(grainB1)):
            point = [grainB1[i][0],grainB1[i][1],grainB1[i][2]]
            min_dist = 1e5
            if len(grainA_old)>0:
                for j in range(len(grainA_old)):
                    b_p = np.array(point)
                    a_p = grainA_old[j]
                    dist_coord = b_p-a_p[:3]
                    dist = la.norm(b_p-a_p[:3])
                    
                    if min_dist>dist:
                        #print(dist)
                        atom_count= a_p[3]
                        index = j
                        min_dist = dist
                grainA_old.pop(index)
                point.append(atom_count)
                gB.append(point)
                countb +=1
        #print(countb,len(grainB1),grainB1[-1],grainB1[-2])
        
        if diag_plt == True:
            diagnostic_plotting(gA,gB, -15,15,-20,20)
    else:
        
        for i in range(len(gB_1)):
            # For m = 1
            if gB_1[i,0]<x_pos+h-1*eps or ((gB_1[i,1]<start or gB_1[i,1]>stop-0*y_eps) and gB_1[i,0]<x_pos-eps):
            #if gB_1[i,0]<x_pos+h-0*eps or ((gB_1[i,1]<start or gB_1[i,1]>stop-10*y_eps) and gB_1[i,0]<x_pos-eps):
                disp = 0
                point = np.array([gB_1[i,1],gB_1[i,0]])
                for dp in range(dipole_number):
                    nodes_for_solid_angle = nodes_modified[dp]
                    solidAngle,disp_temp=solidangle_displacement(nImages, nodes_for_solid_angle, period, point, bur)
                    disp += disp_temp
                if gB_1[i,1]-disp>box[1,1]:
                    disp += 2*box[1,1]
                elif gB_1[i,1]-disp<-box[1,1]:
                    disp -= 2*box[1,1]
                row = np.array([gB_1[i,0],gB_1[i,1]-disp,gB_1[i,2],gB_1[i,3]])
                gB.append(row)
            else:
                disp = 0
                point = np.array([gB_1[i,1],gB_1[i,0]])
                for dp in range(dipole_number):
                    nodes_for_solid_angle = nodes_modified[dp]
                    solidAngle,disp_temp=solidangle_displacement(nImages, nodes_for_solid_angle, period, point, bur)
                    disp += disp_temp
                if gB_1[i,1]-disp>box[1,1]:
                    disp += 2*box[1,1]
                elif gB_1[i,1]-disp<-box[1,1]:
                    disp -= 2*box[1,1]
                grainB_old.append([gB_1[i,0],gB_1[i,1]-disp,gB_1[i,2],gB_1[i,3]])
                
        for i in range(len(gA_1)):
            if gA_1[i,0]>=x_pos-eps or (((gA_1[i,1])>start and (gA_1[i,1])<stop) and gA_1[i,0]>x_pos+h-eps):
                disp = 0
                point = np.array([gA_1[i,1],gA_1[i,0]])
                for dp in range(dipole_number):
                    nodes_for_solid_angle = nodes_modified[dp]
                    solidAngle,disp_temp=solidangle_displacement(nImages, nodes_for_solid_angle, period, point, bur)
                    disp += disp_temp
                if abs(gA_1[i,0]-x_pos-h)<0.2:
                    disp = -2*disp
                if gA_1[i,1]-disp>box[1,1]:
                    disp += 2*box[1,1]
                elif gA_1[i,1]-disp<-box[1,1]:
                    disp -= 2*box[1,1]
                row = np.array([gA_1[i,0],gA_1[i,1]-disp,gA_1[i,2],gA_1[i,3]])
                gA.append(row)
        #print(len(grainB_old),len(grainA1)) 
        if diag_plt == True:
            diagnostic_plotting(grainA1,grainB_old, -20, 20,-20,20)
        
        
        # Add atoms to grain B due to transformation
        for i in range(len(grainA1)):
            point = [grainA1[i][0],grainA1[i][1],grainA1[i][2]]
            disp = 0
            if abs(grainA1[i][0]-x_pos-h)<0.1:
                p = np.array([grainA1[i][1],grainA1[i][0]])
                for dp in range(dipole_number):
                    nodes_for_solid_angle = nodes_modified[dp]
                    solidAngle,disp_temp=solidangle_displacement(nImages, nodes_for_solid_angle, period, p, bur)
                    disp += disp_temp
                disp = -2*disp
            point = [grainA1[i][0],grainA1[i][1]-disp,grainA1[i][2]]
            min_dist = 1e5
            for j in range(len(grainB_old)):
                b_p = np.array(point)
                a_p = grainB_old[j]
                dist_coord = b_p-a_p[:3]
                dist = la.norm(b_p-a_p[:3])
                
                if min_dist>dist:
                    #print(dist)
                    atom_count= a_p[3]
                    index = j
                    min_dist = dist
            grainB_old.pop(index)
            point.append(atom_count)
            gA.append(point)
        
    '''
    gA = grainA1
    gB = grainB1
    '''
    #print(len(gA),len(gB),len(gA)+len(gB))
    gA1 = np.array(gA)
    gB1 = np.array(gB)
    #print(bur,h)
    return gA1,gB1,box
