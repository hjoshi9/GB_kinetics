import numpy as np
import matplotlib.pyplot as plt
from src.IO import read_neb_output_data,write_LAMMPSoutput_tstep

def post_process_neb_data(elem,sigma,misorientation,inclination,size,partitions,disc_mode,steps,folder,outfolder):
    pot_eng = []
    for k in range(steps):
        for j in range(partitions):
            file = folder + "dump.neb_minimized_neigh_Cus"+str(sigma)+"_size"+str(size)+"_step"+str(k)+"."+str(j+1)
            print(file)
            data = read_neb_output_data(file,2)
            tstep = partitions*k+j
            box = data[-1][2]
            Atoms = data[-1][3]
            natoms = Atoms.shape[0]
            pe = np.sum(Atoms[:,5])
            area = (box[1,1]-box[1,0])*(box[2,1]-box[2,0])/100 #nm^2
            pot_eng.append([tstep,pe,area])
            file = "dump.Cu_s"+str(sigma)+"_size"+str(size)+"_"+disc_mode+"_images"+str(partitions)+"_neb"
            message = "Writing output file: " + folder + file
            #print(message)
            f = write_LAMMPSoutput_tstep(natoms,Atoms,box,outfolder,file,tstep)
    peng = np.array(pot_eng)

    out_name = outfolder +elem +"_Sigma"+str(sigma)+"_Misorientation"+str(misorientation)+"_size"+str(size)+"_"+disc_mode+"_parts"+str(partitions)+".txt"
    np.savetxt(out_name,peng)

    # plotting
    plt.figure(dpi=300)
    plt.plot(peng[:,0]/len(peng),(peng[:,1]-peng[0,1])/area,label=disc_mode)
    plt.grid()
    plt.xlabel("Reaction coordinate")
    plt.ylabel(r"$\Delta$E/a (eV/nm$^2$)")
    plt.legend()
    figname = outfolder +elem +"_Sigma"+str(sigma)+"_Misorientation"+str(misorientation)+"_size"+str(size)+"_"+disc_mode+"_parts"+str(partitions)+".png"
    print(figname)
    plt.savefig(figname)

def post_process_gridsearch_data():
    pass

