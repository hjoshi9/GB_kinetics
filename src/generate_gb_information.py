import numpy as np
from src.oilab import find_ATGB_data


def gb_props(sigma,mis,axis,lat_par):
    """
    Reads input parameters and generates the requisit gb properties (burgers vector, step height, period)
    """
    folder = "data/" 
    path = folder + "fcc0-10.txt"
    period_cutoff = 10
    atgb_data = find_ATGB_data(sigma,mis,path,period_cutoff,axis)
    gb_data = atgb_data[0,:]
    # GB properties
    p = atgb_data[0,3]*lat_par
    b = atgb_data[0,4]*lat_par
    h = atgb_data[0,7]*lat_par
    H = atgb_data[0,8]*lat_par
    print("The GB to be investigated is:\n")
    print("Sigma = " + str(gb_data[0]) + "; Misorientation = " + str(gb_data[1]) + "; Inclination = " + str(gb_data[2]) + "\n")
    print("The relevant gb properties are:\n")
    print("period = " +str(p))
    print("smallest burgers vector (b) = "+str(b))
    print("fundamental step height (h) = "+str(h))
    print("distance between CSL planes (H) = "+str(H))
    print("There exist infinetly many disconnection modes with the form (b,h) = (m*b,m*h+n*H), where m and n are integers")
    print("Example of a few of the possible disconnection modes are:")
    print("m \t n \t burgers_vector \t step_height")
    for i in range(-2,3,1):
        for j in range(-2,3,1):
            if i == 0 and j == 0:
                continue
            br = i*b
            sh =  i*h+j*H
            print(str(i) + "\t" + str(j) +"\t" + str(np.round(br,2)) +"\t" + str(np.round(sh,2)))
    m = int(input("Enter the m corrresponding to the disconnection mode:"))
    n = int(input("Enter the n corresponding to the disconnection mode:"))
    bur = m*b
    step_height = m*h+n*H
    return gb_data,bur,step_height
    

