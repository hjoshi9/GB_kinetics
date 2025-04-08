import numpy as np

def write_minimization_input(elem,sigma,mis,inc,lat_par,size,folder,file_name,min_outputfile,dispy,dispz,potential_path):
    """
    Writes lammps file to minimize a system
    """
    min_file = "min.in"
    file = folder + min_file
    #min_output = file_name + "_min"
    min_output_movie = file_name + "_minmov"
    images = 2*size+1
    f = open(file,"w")
    
    f.write("#Minimization of Sigma%d disconnections using LAMMPS\n"%(sigma))
    #f.write("variable s loop %d\n"%(images))
    #f.write("label loops\n")
    f.write("\n")
    
    f.write("#------------------General settings for simulation----------------------------\n")
    f.write("clear\n")
    f.write("units metal\n")
    f.write("dimension 3\n")
    f.write("boundary m p p\n")
    f.write("atom_style atomic\n")
    f.write("neighbor	0.3 bin\n")
    f.write("neigh_modify	delay 5\n")
    f.write("atom_modify  map array sort 0 0.0\n")
    #f.write("echo both\n")
    f.write("\n")
    
    f.write("#----------------- Variable declaration---------------------------------------\n")
    f.write("variable a equal %f\n"%(lat_par))
    f.write("variable potpath string %s\n"%(potential_path))
    f.write("variable sigma equal %d\n"%(sigma))
    f.write("variable y_image equal %d\n"%(size))
    f.write("variable dispy equal %f\n"%dispy)
    f.write("variable dispz equal %f\n"%dispz)
    #f.write("variable step equal $s-1\n")
    f.write("variable elem string %s\n"%(elem))
    f.write("variable mis equal %d\n"%(mis))
    f.write("variable folder string %s\n"%(folder))
    f.write("variable file_name string ${folder}%s\n"%(file_name))
    #f.write("variable file_name string ${folder}/data.${elem}s${sigma}inc0.0_size_${y_image}disc_new${step}\n")
    f.write("variable out_file1 string ${folder}%s\n"%(min_outputfile))
    #f.write("variable out_file2 string ${folder}/data.${elem}s${sigma}_size${y_image}_min_d${step}\n")
    f.write("variable out_file_mov string ${folder}%s\n"%(min_output_movie))
    f.write("#----------------------- Atomic structure ----------------------\n")
    f.write("lattice fcc $a\n")
    f.write("read_data ${file_name}\n")
    f.write("group upper type 1\n")
    f.write("group lower type 2\n")
    f.write("\n")
    
    f.write("#----------------------- InterAtomic Potential --------------------\n")
    f.write("pair_style eam/alloy\n")
    f.write("pair_coeff * * ${potpath} %s %s\n"%(elem,elem))
    f.write("neighbor 2 bin\n")
    f.write("neigh_modify delay 10 check yes\n")
    
    f.write("#---------- Displace top part for lowest energy structure ----------\n")
    f.write("delete_atoms overlap 1 upper upper\n")
    f.write("delete_atoms overlap 1 lower lower\n")
    f.write("delete_atoms overlap 0.1 upper lower\n")
    f.write("displace_atoms upper move 0 ${dispy} ${dispz} units box\n")
    f.write("\n")
    
    f.write("# Apply fix to tether centroid of the system to the center\n")
    f.write("fix pull all spring tether 10.0 0.0 0.0 0.0 0.0 \n")
    
    f.write("#--------------------- Minimize ------------------------------------\n")
    f.write("thermo 250\n")
    f.write("thermo_style custom step temp pe lx ly lz press pxx pyy pzz\n")
    f.write("dump 1 all custom 1000 ${out_file_mov} id type x y z\n")
    f.write("min_style cg\n")
    f.write("minimize 1e-25 1e-25 10000 10000\n")
    '''
    f.write("#-------- Minimize with box relaxation in y direction -------------\n")
    f.write("fix 1 all box/relax y 0 vmax 0.001\n")
    f.write("min_style cg\n")
    f.write("minimize 1e-25 1e-25 5000 5000\n")
    
    f.write("#------- Minimize with box relaxation in z direction ---------------\n")
    f.write("fix 1 all box/relax z 0 vmax 0.001\n")
    f.write("min_style cg\n")
    f.write("minimize 1e-25 1e-25 5000 5000\n")
    
    f.write("#--------------------- Minimize ------------------------------------\n")
    f.write("min_style cg\n")
    f.write("minimize 1e-25 1e-25 10000 10000\n")
    '''
    f.write("write_data ${out_file1}\n")
    
    f.close()
    
    return file

def create_fix_eco_orientationfile(sigma,mis,inc,folder,a,b,lat_par):
    """
    Writes eco orient grain orientation files, in case MD simulations are to be done for a system
    """
    first_grain = a.T*lat_par
    second_grain = b.T*lat_par
    #print(first_grain)
    #print(second_grain)
    file = "Sigma"+str(sigma)+"_mis"+str(mis)+"_inc"+str(inc)+".ori"
    f = open(folder+file,"w")
    for i in range(first_grain.shape[0]):
        f.write("%f %f %f\n"%(first_grain[i,0],first_grain[i,1],first_grain[i,2]))
    for i in range(second_grain.shape[0]):
        f.write("%f %f %f\n"%(second_grain[i,0],second_grain[i,1],second_grain[i,2]))
    f.close()
    return folder+file

def write_lammps_gridsearch_input(folder,infile,outfolder,potential,elem,lat_par,sigma,size,mis,step_increments,limit,output_setting=0):
    """
    Writes lammps file to conduct a grid search
    """
    file = "grid_search.in"
    file = folder + file
    outfile = elem+"_"+"sigma"+str(sigma)+"_mis"+str(mis)+"_size"+str(size)+"_gridsearch_results.txt"
    yloop = int(2*limit*step_increments) + 1
    zloop = yloop
    yloop0 = int(limit*step_increments)
    zloop0 = yloop0
    f = open(file,"w")
    f.write("variable disp_county loop %d\n"%yloop)
    f.write("label loopy\n")
    f.write("variable disp_countz loop %d\n"%zloop)
    f.write("label loopz\n")
    f.write("variable a equal %f\n"%lat_par)
    f.write("variable potpath string %s\n"%potential)
    f.write("variable sigma equal %d\n"%sigma)
    f.write("variable y_image equal %d\n"%size)
    f.write("variable dispy equal %f*(${disp_county}-%d)\n"%(step_increments,yloop0))
    f.write("variable dispz equal %f*(${disp_countz}-%d)\n"%(step_increments,zloop0))
    f.write("variable step equal 0\n")
    f.write("variable elem string %s\n"%elem)
    f.write("variable mis equal %d\n"%mis)
    f.write("variable folder string %s\n"%folder)
    f.write("variable file_name string ${folder}/%s \n"%infile)
    if output_setting == 1:
        f.write("variable outfile2 string ${folder}/%s_dy${dispy}dz${dispz}\n"%infile)
    f.write("variable outfile string ${folder}/%s\n"%outfile)
    f.write("clear\n")
    f.write("units metal\n")
    f.write("dimension 3\n")
    f.write("boundary m p p\n")
    f.write("atom_style atomic\n")
    f.write("echo both\n")
    f.write("neighbor        0.3 bin\n")
    f.write("neigh_modify    delay 1\n")
    f.write(" atom_modify  map array sort 0 0.0\n")
    f.write("lattice fcc $a\n")
    f.write("read_data ${file_name}\n")
    f.write("replicate 1 1 2\n")
    f.write("group upper type 1\n")
    f.write("group lower type 2\n")
    f.write("variable midbox equal xhi/2+xlo/2\n")
    f.write("variable gb_thickness equal 10\n")
    f.write("variable gblo equal ${midbox}-${gb_thickness}\n")
    f.write("variable gbhi equal ${midbox}+${gb_thickness}\n")
    f.write("variable bulklo equal ${midbox}+1.2*${gb_thickness}\n")
    f.write("variable bulkhi equal xhi-0.25*${gb_thickness}\n")
    f.write("region BULK block ${bulklo} ${bulkhi} INF INF INF INF units box\n")
    f.write("region GB block ${gblo} ${gbhi} INF INF INF INF units box\n")
    f.write("variable area equal ly*lz\n")
    f.write("pair_style eam/alloy\n")
    f.write("pair_coeff * * ${potpath} ${elem} ${elem}\n")
    f.write("neighbor 2 bin\n")
    f.write("neigh_modify delay 10 check yes\n")
    f.write("compute eng all pe/atom\n")
    f.write("compute eatoms all reduce sum c_eng\n")
    f.write("compute csym all centro/atom fcc\n")
    f.write("group           GB region GB\n")
    f.write("group           BULK region BULK\n")
    f.write("compute         peratom GB pe/atom\n")
    f.write("compute         peratombulk BULK pe/atom\n")
    f.write("compute         pe GB reduce sum c_peratom\n")
    f.write("compute         pebulk BULK reduce sum c_peratombulk\n")
    f.write("variable        peGB equal c_pe\n")
    f.write("variable        peBULK equal c_pebulk\n")
    f.write("variable        atomsGB equal count(GB)\n")
    f.write("variable        atomsBULK equal count(BULK)\n")
    f.write("delete_atoms overlap 1 upper upper\n")
    f.write("delete_atoms overlap 1 lower lower\n")
    f.write("delete_atoms overlap 0.1 upper lower\n")
    f.write("displace_atoms upper move 0 ${dispy} ${dispz} units box\n")
    f.write("thermo 500\n")
    f.write("thermo_style custom step temp pe lx ly lz press pxx pyy pzz c_eatoms\n")
    if output_setting == 1:
        f.write("dump            1 all custom 5000 ${outfile2} id type x y z c_csym c_eng\n")
    f.write("variable pot_eng equal pe\n")
    f.write("\n")
    f.write("min_style cg\n")
    f.write("minimize 1e-25 1e-25 5000 5000\n")
    f.write("\n")
    f.write("variable        coh equal (${peBULK}/${atomsBULK})\n")
    f.write("variable        conversion_factor equal 16.02\n")
    f.write("variable        GBene equal (${conversion_factor}*(${peGB}/${area}-${coh}*${atomsGB}/${area}))\n")
    f.write("print \"dispy = ${dispy} dispz = ${dispz} GBene = ${GBene}\" append ${outfile}\n")
    f.write("next disp_countz\n")
    f.write("jump SELF loopz\n")
    f.write("next disp_county\n")
    f.write("jump SELF loopy\n")
    f.write("\n")
    f.write("\n")
    f.close()
    
    return outfile,file
