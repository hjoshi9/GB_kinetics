import numpy as np

def read_data(path_r,sigma_val):
    """
    Function to read the data file containing the misorientation, inlclination
    burgers vectors and step heights
    Input : path_  = path of .txt file
    Output : Array data with row format : {sigma,mis,inc,period,tn_x,tn_y,tn_z,bn_x,bn_y,bn_z,bx,by,bz,h,bx1,by1,bz1,h1,bx2,by2,bz2,h2,H}
    """
    i = 0
    q = -1
    j = q
    data = []
    k=0
    with open(path_r,'r') as file:
        flag = 0
        
        for line in file:
            fields = line.split(' ')
            i = i+1
            if(len(fields)==3 and fields[0] == "Sigma"):
                j=-1
                flag = 1
                row = []
                sig = (float(fields[2]))

            if(fields[0] == "Misorientation="):
                n = len(fields)
                mis = (float(fields[n-1]))
                
            if (len(fields)>15): 
                nrow =[]
                if fields[2] != "Inclination" and fields[1] != "-------------":
                    flag = 2
                    for t in range(len(fields)):
                        if fields[t]!="":
                            j = j+1
                            nrow.append(fields[t])
                    
            if flag == 2:# and sig == sigma_val:        
                row = [sig,mis,float(nrow[0]),float(nrow[7]),float(nrow[1]),float(nrow[2]),float(nrow[3]),float(nrow[4]),
                       float(nrow[5]),float(nrow[6]),float(nrow[8]),float(nrow[9]),float(nrow[10]),float(nrow[14]),float(nrow[16]),
                       float(nrow[17]),float(nrow[18]),float(nrow[22]),float(nrow[23]),
                       float(nrow[24]),float(nrow[25]),float(nrow[29]),float(nrow[15]),float(nrow[11]),float(nrow[12]),float(nrow[13])]
                #print(nrow)
                data.append(row)
    return data

def find_ATGB_data(sigma,mis,path,period_cutoff,axis):
    """
    Function to generate the atgb_data to be obtained from SNF input file
    Input : (sigma number, misorientation, inout file path, maximum CSL period required)
    Output:
    0:sigma_num, 1:misorientation, 2:inclination, 3:period, 4:burgers_vector_x, 5:burgers_vector_y,
    6:burgers_vector_z, 7:step height, H, 8:step_vector_x, 9:step_vector_y, 10:step_vector_z 
    """
    data = np.asarray(read_data(path,1))
    #print(data)
    atgb_data = []
    prev_inc = -10 
    for j in range(len(data)):
        if  abs(data[j,1]-mis)<0.2 and abs(data[j,2]-prev_inc)>3 and data[j,3]<period_cutoff:
            atgb_data.append([data[j,0],data[j,1],data[j,2],data[j,3],data[j,10],data[j,11],data[j,12],data[j,13],data[j,22],data[j,23],data[j,24],data[j,25]])
            prev_inc = data[j,2]
        if abs(data[j,1]-mis)<0.2 and data[j,2]==45.0:
            atgb_data.append([data[j,0],data[j,1],data[j,2],data[j,3],data[j,10],data[j,11],data[j,12],data[j,13],data[j,22],data[j,23],data[j,24],data[j,25]])
    max_inc_onfile = atgb_data[-1][2]
    # Extend the range from current to 90 degrees
    for j in range(len(atgb_data)):
        if axis[0]==0 and axis[1]==0 and axis[2]==1:
            if 45+atgb_data[j][2]>max_inc_onfile and 45+atgb_data[j][2]<=90:
               atgb_data.append([atgb_data[j][0],atgb_data[j][1],45+atgb_data[j][2],atgb_data[j][3],data[j][10],data[j][11],data[j][12],data[j][13],data[j][22],data[j,23],data[j,24],data[j,25]]) 
        if axis[0]==1 and axis[1]==1 and axis[2]==1:
            if 60+atgb_data[j][2]>max_inc_onfile and 60+atgb_data[j][2]<=90:
               atgb_data.append([atgb_data[j][0],atgb_data[j][1],60+atgb_data[j][2],atgb_data[j][3],data[j][10],data[j][11],data[j][12],data[j][13],data[j][22],data[j,23],data[j,24],data[j,25]]) 
    # Sort the atgb data w.r.t inclination
    a = np.array(atgb_data)
    col = 2
    a=a[np.argsort(a[:,col])]
    
    return a
