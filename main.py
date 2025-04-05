import numpy as np

from src.create_disconnections import *
from src.generate_gb_information import *

folder = "data/"
path = folder + "fcc0-10.txt"
elem = "Cu"
sigma = 13
mis = 22.62
lat_par = 3.615
axis = [0,0,1]
size = 2
lat_Vec = np.array([[0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.5],
                    [0.5, 0.0, 0.5]])
reg_parameter = 0.01
max_iters = 500000
dipole_number = 3
# Figure out disconnection information
gb_data,bur,step_height = gb_props(sigma,mis,path,axis,lat_par)

# Create disconnection images
a = generate_disconnection_images(gb_data,bur,step_height,lat_par,lat_Vec,axis,size,elem,reg_parameter,max_iters)