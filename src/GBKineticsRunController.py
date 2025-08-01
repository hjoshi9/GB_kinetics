# This module only defines the generate_disconnection_images function.
# It is not intended to be run directly.
import numpy as np
import os
from src.bicrystal import bicrystal
from src.IO import *
from src.runLAMMPS import run_LAMMPS
from src.min_shuffle import min_shuffle
from src.bicrystal import bicrystallography


def runGBkinetics(sig, mis, inc, lat_par, lat_Vec, axis, size_y, size_z, elem, reg_parameter,
                max_iters, lammps_location, mpi_location, folder, potential,
                  dispy, dispz, oilab_output_file,choose_disconnection=True,run_neb = False):
    # Bicrystallographic GB properties
    bc = bicrystallography(sig,mis,inc, axis, lat_par)
    gb_data, bur, step_height = bc.gb_props(oilab_output_file, choose_disconnection)
    sigma = int(gb_data[0])
    mis = np.round(gb_data[1])
    inc = gb_data[2]
    p = gb_data[3] * lat_par

    # Box lengths
    non_periodic_direction_size = 100
    size_along_period = size_y
    size_along_tilt_axis = size_z

    # Geometric GB properties
    offset = 1 * p / 16
    gb_position = 0
    section_factor = 2
    total_images = int(size_along_period * section_factor) + 1

    # Define output folder
    out_folder = folder + elem + "/Sigma" + str(int(sigma)) + "/Misorientation" + str(np.round(mis)) + "/size" + str(
        size_along_period) + "/b" + str(np.round(bur, 2)) + "h" + str(np.round(step_height, 2)) + "/"
    os.makedirs(out_folder, exist_ok=True)

    # Decision variables for testing and isolated runs
    create_bicrystal_decision = True  # or False
    min_decision = False  # or False
    min_shuffle_decision = False  # or False
    create_eco_input = False

    # Define a bicrystal
    Bicrystal = bicrystal(gb_data,axis,lat_par,lat_Vec,size_along_period,size_along_tilt_axis,non_periodic_direction_size)
    Bicrystal._setup_bicrystal()

    # Define LAMMPS handler
    run_lmp = run_LAMMPS(out_folder, elem, lat_par, sigma, mis, inc, size_along_period,potential, mpi_location, lammps_location)

    # Define minshuffle operator
    min_shuffle_operator = min_shuffle(lat_par,sigma,mis,inc,p,out_folder,elem,reg_parameter,max_iters)

    # Create bicrsytals with GB motion mediated with disconnection mediated migration
    for image_num in range(total_images):
        # Define dislocation locations forming the disconnection
        perturb = 0.1
        if image_num < total_images:
            nodes = np.array([[-image_num * p / section_factor + offset - perturb, step_height + gb_position],
                              [ image_num * p / section_factor + offset + perturb, step_height + gb_position]])
        else:
            fac = 4
            nodes = np.array([[-image_num * p / section_factor - p / 2 * fac , step_height + gb_position],
                              [ image_num * p / section_factor + p / 2 * fac , step_height + gb_position]])
        disloc1 = nodes[0]
        disloc2 = nodes[1]
        # Create and minimize bicrystals
        if image_num == 0:
            if create_bicrystal_decision == True:
                print("\n============= Generating initial flat GB  ==================")
                Bicrystal.create_flat_gb_bicrystal(gb_position)
                file_name_init = Bicrystal.ordered_write(out_folder, elem, image_num, gb_position)
            if min_decision == True:
                min_outputfile_init = run_lmp.run_minimization(file_name_init, dispy, dispz)
        else:
            if create_bicrystal_decision == True:
                print("\n================= Generating GB image " + str(image_num) + " bicrystallographically ==============")
                Bicrystal.create_disconnection_containing_bicrystal(nodes,bur,step_height,gb_position,image_num)
                file_name_image = Bicrystal.ordered_write(out_folder, elem, image_num, gb_position, step_height,disloc1[0],disloc2[0])
            if min_decision == True:
                min_outputfile_image = run_lmp.run_minimization(file_name_image, dispy, dispz)

            if min_shuffle_decision == True:
                print("==================== Generating atomic trajectories =========================")
                # Apply min shuffle
                if min_decision == True:
                    file_mode = 2
                    file_flat = out_folder + min_outputfile_init
                    file_disconnection = out_folder + min_outputfile_image
                else:
                    file_mode = 1
                    file_flat = out_folder + file_name_init
                    file_disconnection = out_folder + file_name_image
                min_shuffle_operator.load_data(file_mode,file_flat,file_disconnection)
                min_shuffle_operator.gb_info(file_flat+"mov", step_height, disloc1, disloc2)
                min_shuffle_operator.format_input(out_folder)
                min_shuffle_operator.run()
                min_shuffle_operator.write_images(out_folder,image_num)
                if run_neb == True:
                    min_shuffle_operator._write_neb_input_file(out_folder, image_num)
    #run_lmp.run_neb_calc(burgers_vector, step_height)
    # %%
    if create_eco_input == True:
        Bicrystal.create_fix_eco_orientationfile(out_folder)

    return out_folder

def runGridSearch(sig, mis, inc, lat_par, lat_Vec, axis, size_y, size_z, elem, lammps_location,
                  mpi_location, folder, potential, oilab_output_file, choose_disconnection = False):

    # Bicrystallographic GB properties
    bc = bicrystallography(sig, mis, inc, axis, lat_par)
    gb_data, bur, step_height = bc.gb_props(oilab_output_file,choose_disconnection)
    sigma = int(gb_data[0])
    mis = np.round(gb_data[1])
    inc = gb_data[2]
    p = gb_data[3] * lat_par

    # Box lengths
    non_periodic_direction_size = 100
    size_along_period = size_y
    size_along_tilt_axis = size_z

    # Geometric GB properties
    gb_position = 0

    # Define output folder
    out_folder = folder + elem + "/Sigma" + str(int(sigma)) + "/Misorientation" + str(np.round(mis)) + "/grid_search/"
    os.makedirs(out_folder, exist_ok=True)

    # Define a bicrystal
    Bicrystal = bicrystal(gb_data, axis, lat_par, lat_Vec, size_along_period, size_along_tilt_axis,non_periodic_direction_size)
    A, B, ga, gb = Bicrystal._setup_bicrystal()

    # Define LAMMPS handler
    run_lmp = run_LAMMPS(folder, elem, lat_par, sigma, mis, inc, size_along_period, potential, mpi_location,lammps_location)

    # Create a flat bicrystal
    print("\n============= Generating initial flat GB  ==================")
    gA_1, gB_1, box1 = Bicrystal.create_flat_gb_bicrystal(gb_position)
    name_suffix = "size_" + str(size_along_period)
    file_name_init = Bicrystal.ordered_write(out_folder, elem, name_suffix, gb_position)

    # Run grid search
    print("\n================= Running grid search  =====================")
    out = run_lmp.run_neb_calc(file_name_init, out_folder)

    return out_folder
