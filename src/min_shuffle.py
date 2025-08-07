import numpy as np
import numpy.linalg as la
import ot
import ot.plot
import matplotlib.pyplot as plt
from src.IO import *

class min_shuffle:
    def __init__(self,lattice_parameter,sigma,misorientation,inclination,period,folder,elem,reg_param,max_iters,box_expansion_factor=0,dimension=3):
        self.lattice_parameter = lattice_parameter
        self.sigma = sigma
        self.misorientation = misorientation
        self.inclination = inclination
        self.period = period
        self.folder = folder
        self.element = elem
        self.reg_param = reg_param
        self.max_iters = max_iters
        self.box_expansion_factor = box_expansion_factor
        self.dimension = dimension

        self.grainA_pretransform = None
        self.grainB_pretransform = None
        self.grainA_posttransform = None
        self.grainB_posttransform = None
        self.initial_grain = None
        self.final_grain = None
        self.box = None

        self.gb_location = None
        self.step_height = None
        self.dislocation_start = None
        self.dislocation_end = None

        self.atoms_minshuf = None
        self.types_minshuf = None
        self.box_minshuf = None

        self.initial_atoms_transformed_region = None
        self.final_atoms_transformed_region = None

    def gb_info(self,filepath,st_height,disloc1,disloc2,min_shuffle_domain_expansion_factor=1.5,disconnection_extension=5):
        lat_par = self.lattice_parameter
        p = self.period
        average_gb_loc, gb_hi, gb_lo = find_gb_location(filepath)
        if st_height > 0:
            gb_location = 1 * average_gb_loc - 0.5 * lat_par * p / 16
        else:
            gb_location = 1 * average_gb_loc + 0.5 * lat_par * p / 16
        d_start = disloc1[0] - disconnection_extension
        d_stop = disloc2[0] + disconnection_extension
        self.gb_location = gb_location
        self.step_height = st_height*(1+ min_shuffle_domain_expansion_factor)
        self.dislocation_start = d_start
        self.dislocation_end = d_stop


    def load_data(self,file_mode,file_flat,file_disconnection):
        print("=============================== Generating atomic trajectories =====================================")
        data_init = read_LAMMPS_datafile(file_flat, file_mode)
        data_final = read_LAMMPS_datafile(file_disconnection, file_mode)
        atoms = data_init[0][3]
        box = data_init[0][2]
        Ai = []
        Bi = []
        neb_init = []
        for i in range(atoms.shape[0]):
            if atoms[i, 1] == 1.0:
                Ai.append([atoms[i, 2], atoms[i, 3], atoms[i, 4], atoms[i, 0]])
            # elif atoms
            elif atoms[i, 1] == 2.0:
                Bi.append([atoms[i, 2], atoms[i, 3], atoms[i, 4], atoms[i, 0]])
            else:
                neb_init.append([atoms[i, 2], atoms[i, 3], atoms[i, 4], atoms[i, 0]])
        Ai = np.asarray(Ai)
        Bi = np.asarray(Bi)
        initial = atoms

        atoms = data_final[0][3]
        Af = []
        Bf = []
        neb_final = []
        for i in range(atoms.shape[0]):
            if atoms[i, 1] == 1.0:
                Af.append([atoms[i, 2], atoms[i, 3], atoms[i, 4], atoms[i, 0]])
            elif atoms[i, 1] == 2.0:
                Bf.append([atoms[i, 2], atoms[i, 3], atoms[i, 4], atoms[i, 0]])
            else:
                neb_final.append([atoms[i, 2], atoms[i, 3], atoms[i, 4], atoms[i, 0]])
        Af = np.asarray(Af)
        Bf = np.asarray(Bf)
        final = atoms

        self.grainA_pretransform = Ai
        self.grainB_pretransform = Bi
        self.grainA_posttransform = Af
        self.grainB_posttransform = Bf
        self.box = box
        self.initial_grain = initial
        self.final_grain = final

    def format_input(self,folder):
        if self.step_height is None:
            raise ValueError("GB details not defined. Run gb_info()")
        if self.grainA_posttransform is  None:
            raise ValueError("Data not loaded. Run load_data()")
        Ai = self.grainA_pretransform
        Bi = self.grainB_pretransform
        Af = self.grainA_posttransform
        Bf = self.grainB_posttransform
        box = self.box
        gb_loc =self.gb_location
        h = self.step_height
        disc_start = self.dislocation_start
        disc_end = self.dislocation_end
        elem = self.element
        sigma = self.sigma
        inc = self.inclination

        data_A = []
        data_B = []
        eps = 1e-1
        count = 1
        if h >= 0:
            initial = np.concatenate((Ai, Bi), axis=0)
            for i in range(initial.shape[0]):
                if initial[i, 0] < gb_loc + h - eps and initial[i, 0] > gb_loc - eps and initial[i, 1] >= disc_start and initial[i, 1] <= disc_end:
                    count += 1
                    data_A.append([initial[i, 0], initial[i, 1], initial[i, 2], initial[i, 3]])
        else:
            for i in range(Bi.shape[0]):
                if Bi[i, 0] > gb_loc + h - eps and Bi[i, 0] < gb_loc - eps and Bi[i, 1] >= disc_start and Bi[i, 1] <= disc_end:
                    count += 1
                    data_A.append([Bi[i, 0], Bi[i, 1], Bi[i, 2], Bi[i, 3]])
        A = np.array(data_A)
        for i in range(Af.shape[0]):
            for j in range(len(data_A)):
                if abs(Af[i, 3] - A[j, 3]) < 0.5:
                    count += 1
                    data_B.append([Af[i, 0], Af[i, 1], Af[i, 2], Af[i, 3]])
                    break

        for i in range(Bf.shape[0]):
            for j in range(len(data_A)):
                if abs(Bf[i, 3] - A[j, 3]) < 0.5:
                    count += 1
                    data_B.append([Bf[i, 0], Bf[i, 1], Bf[i, 2], Bf[i, 3]])
                    break
        B = np.array(data_B)
        # Write to data file
        suffix = "min_shuffle"
        file = "data." + elem + "s" + str(sigma) + "inc" + str(inc)
        outfile = write_lammps_datafile(folder, file, suffix, A, B, box)
        # Prepare output
        atoms = np.concatenate((A, B), axis=0)
        types = 2
        eps2 = 0.1
        xlo = gb_loc - h + eps2
        xhi = gb_loc + h + eps2
        scale = 1.05
        box = np.array([[scale * xlo, scale * xhi],
                        [box[1, 0], box[1, 1]],
                        [box[2, 0] + 1e-2, box[2, 1]]])
        self.atoms_minshuf = np.zeros((len(A)+len(B),5))
        for i in range(len(A)):
            self.atoms_minshuf[i,0] = A[i,3]
            self.atoms_minshuf[i,1] = 1
            self.atoms_minshuf[i,2:] = A[i,:3]
        for i in range(len(B)):
            self.atoms_minshuf[i+len(A), 0] = B[i,3]
            self.atoms_minshuf[i+len(A), 1] = 2
            self.atoms_minshuf[i+len(A), 2:] = B[i, :3]
        self.types_minshuf = types
        self.box_minshuf = box

    def replicate(self,rep_scheme,X,Y,box):
        dim = self.dimension
        # Define outputs
        rep_box = 0 * box

        # Define multipliers
        xlo_mult = rep_scheme[0]
        xhi_mult = rep_scheme[1]
        ylo_mult = rep_scheme[2]
        yhi_mult = rep_scheme[3]
        zlo_mult = rep_scheme[4]
        zhi_mult = rep_scheme[5]

        x_range = range(xlo_mult, xhi_mult + 1)
        y_range = range(ylo_mult, yhi_mult + 1)
        z_range = range(zlo_mult, zhi_mult + 1)
        nx = len(x_range)
        ny = len(y_range)
        nz = len(z_range)

        # Find lenght of box along each of the axes
        Lx = box[0, 1] - box[0, 0]
        Ly = box[1, 1] - box[1, 0]
        Lz = box[2, 1] - box[2, 0]

        nx = len(X)
        ny = len(Y)

        X_rep = np.zeros((X.shape[0] * nx * ny * nz, dim))
        Y_rep = np.zeros((Y.shape[0] * nx * ny * nz, dim))
        # Replicate
        ind_count = 0
        for i in x_range:
            for j in y_range:
                for k in z_range:
                    x_add = i * Lx
                    y_add = j * Ly
                    z_add = k * Lz

                    x_new = X + np.array(x_add, y_add, z_add)
                    y_new = Y + np.array(x_add, y_add, z_add)

                    start = ind_count * nx
                    end = (ind_count + 1) * nx
                    X_rep[start:end, :] = x_new
                    Y_rep[start:end, :] = y_new
                    ind_count += 1

        # Update box
        rep_box[0, 0] = box[0, 0] + xlo_mult * Lx
        rep_box[0, 1] = box[0, 1] + xhi_mult * Lx
        rep_box[1, 0] = box[1, 0] + ylo_mult * Ly
        rep_box[1, 1] = box[1, 1] + yhi_mult * Ly
        rep_box[2, 0] = box[2, 0] + zlo_mult * Lz
        rep_box[2, 1] = box[2, 1] + zhi_mult * Lz

        return X_rep, Y_rep, rep_box

    def pbcdist(self,dp,Lx,Ly,Lz):
        dim = self.dimension
        d = dp
        n = dim
        L = np.array([Lx, Ly, Lz])
        for i in range(n):
            Li = L[i]
            for j in range(dp.shape[0]):
                if dp[j, i] < -0.5 * Li:
                    d[j, i] += Li
                elif dp[j, i] >= 0.5 * Li:
                    d[j, i] -= Li
        return d

    def pbcwrap(self,d,box):
        dim = self.dimension
        dpbc = d

        xlo = box[0, 0]
        ylo = box[1, 0]
        zlo = box[2, 0]
        xhi = box[0, 1]
        yhi = box[1, 1]
        zhi = box[2, 1]
        Lx = xhi - xlo
        Ly = yhi - ylo
        Lz = zhi - zlo

        L = np.array([Lx, Ly, Lz])

        for i in range(dim):
            di = d[:, i]
            Li = L[i]
            lo = box[i, 0]
            hi = box[i, 1]
            for j in range(len(d)):
                xj = di[j]
                if xj < lo:
                    while xj < lo:
                        xj += Li
                if xj >= hi:
                    while xj >= hi:
                        xj -= Li
                dpbc[j, i] = xj
        return dpbc

    def run(self):
        # Import input parameters
        reg_param = self.reg_param
        iter_max = self.max_iters
        box_expansion_factor = self.box_expansion_factor
        dim = self.dimension
        #lattice_parameter = self.lattice_parameter  # angstroms
        #r0 = lattice_parameter / np.sqrt(2)  # nearest neighbor distance in perfect FCC crystal

        if self.types_minshuf is None:
            raise ValueError("Input formatting not done yet. Run format_input()")
        atoms = self.atoms_minshuf
        types = self.types_minshuf
        box = self.box_minshuf

        # Find box size and atom positions for each of the types of particles
        N = atoms.shape[0]
        box += box_expansion_factor * np.array([[0, 0], [-5, 5], [0, 0]])
        xlo = box[0, 0]
        ylo = box[1, 0]
        zlo = box[2, 0]
        xhi = box[0, 1]
        yhi = box[1, 1]
        zhi = box[2, 1]
        Lx = xhi - xlo
        Ly = yhi - ylo
        Lz = zhi - zlo

        type_list = [x + 1 for x in range(types)]
        # Find the particles of each of the grains: X is grain 1 Y is grain 2
        Xbasis = []  # type 1
        Ybasis = []  # type 2
        indicies = []
        for i in range(N):
            if atoms[i, 1] == 1:
                Xbasis.append(atoms[i,2:])
                indicies.append(atoms[i, 0])
            elif atoms[i, 1] == 2:
                Ybasis.append(atoms[i,2:])
        Xbasis = np.array(Xbasis)
        Ybasis = np.array(Ybasis)
        # print(Xbasis.shape)
        if Xbasis.shape[0] == 0 or Ybasis.shape[0] == 0:
            raise ValueError("Xbasis and Ybasis must have same non-zero length")
        if Xbasis.shape[0] != Ybasis.shape[0]:
            raise ValueError("The number of atoms do not match for the two grains. One-to-one mapping not possible")
        else:
            N = Xbasis.shape[0]

        # Find displacement vectors with pbcs
        pbcon = True
        dist_mat = np.zeros((N, N))
        for i in range(N):
            dvec = Ybasis - Xbasis[i, :]
            dvec_pbc = self.pbcdist(dvec, Lx, Ly, Lz)
            for j in range(N):
                dist_mat[i, j] = la.norm(dvec_pbc[j, :])

        if pbcon == True:
            dist_mat = dist_mat ** 2

        p = np.zeros(dim)  # Recomputed, set to [0,0,0] if unknown and results will match TDP

        # Sinkhorn's algorithm
        a = np.ones(N) / N
        b = np.ones(N) / N
        Gamma = ot.bregman.sinkhorn_log(a, b, dist_mat, reg_param, iter_max)

        # Recover the displacement vectors from OT matrix(gamma)
        cutoff = 1 / (N * 2)
        gamma_mod = min_shuffle._threshold(Gamma, cutoff)
        I, J = np.where(gamma_mod != 0)
        # I,J = np.where(Gamma!=0)
        maxNp = np.max(gamma_mod)
        disp_vecs = np.zeros((len(I), 11))
        for i in range(len(I)):
            K = gamma_mod[I[i], J[i]]  # path prob
            Xcoords = np.zeros((1, dim))
            Ycoords = np.zeros((1, dim))
            Xcoords[0, :] = np.array([Xbasis[I[i], 0], Xbasis[I[i], 1], Xbasis[I[i], 2]])
            Ycoords[0, :] = np.array([Ybasis[J[i], 0], Ybasis[J[i], 1], Ybasis[J[i], 2]])
            index = indicies[I[i]]

            # Vector connecting X and Y
            disp_vec_new = Ycoords - Xcoords
            if pbcon == True:
                disp_pbc = self.pbcdist(disp_vec_new, Lx, Ly, Lz)
            newYcoords = Xcoords + disp_pbc

            disp_vecs[i, :] = np.array([disp_pbc[0, 0], disp_pbc[0, 1], disp_pbc[0, 2], K, Xcoords[0, 0],
                                        Xcoords[0, 1], Xcoords[0, 2], newYcoords[0, 0], newYcoords[0, 1],
                                        newYcoords[0, 2], index])

        # Data in different frames
        ndisps = disp_vecs.shape[0]
        Kvec = disp_vecs[:, 3]
        Kvecrenorm = la.norm(Kvec)

        Xcoords = disp_vecs[:, 4:7]
        Ycoords = disp_vecs[:, 7:10]
        dTDP = disp_vecs[:, 0:3]
        XTDP = Xcoords
        YTDP = Ycoords
        dCDP = dTDP - p
        XCDP = Xcoords
        YCDP = Ycoords - p
        prob = np.zeros((len(Kvec), 3))
        idx = disp_vecs[:, 10]
        for i in range(len(Kvec)):
            prob[i, :] = Kvec[i] * dTDP[i, :]
        Dvec = np.sum(prob, 0)  # probabilistic expression for total net displacement/atom
        dSDP = dTDP - Dvec
        XSDP = Xcoords
        YSDP = Ycoords - Dvec

        Mvecest = Dvec - p
        microvecs = np.array([p[0], p[1], p[2],
                              Mvecest[0], Mvecest[1], Mvecest[2],
                              Dvec[0], Dvec[1], Dvec[2]])
        Xp = XTDP
        Yp = YTDP

        Xs = np.zeros((ndisps, 5))
        Ys = np.zeros((ndisps, 5))  # copies of coordinates to overwrite with sheared coordinates
        k = 0
        prob_cutoff = 1e-8
        for i in range(ndisps):
            path_prob = Kvec[i]
            if path_prob > prob_cutoff:
                a = np.zeros((1, 3))
                b = np.zeros((1, 3))
                a[0, :] = Xp[i, :]
                b[0, :] = Yp[i, :]
                if pbcon == True:
                    dp_new = self.pbcdist(b - a, Lx, Ly, Lz)
                b_new = a + dp_new
                Xs[k, :3] = a
                Xs[k, 3] = idx[i]
                Xs[k,4] = k
                Ys[k, :3] = b_new
                Ys[k, 3] = idx[i]
                Ys[k,4] = k
                flag = 0
                # Check for PBCs along y and z
                if b_new[0, 1] > yhi:
                    b_new[0, 1] -= Ly
                    flag = 1
                if b_new[0, 2] > zhi:
                    b_new[0, 2] -= Lz
                    flag = 1
                if b_new[0, 1] < ylo:
                    b_new[0, 1] += Ly
                    flag = 1
                if b_new[0, 2] <= zlo + 0.1:
                    b_new[0, 2] += Lz
                    flag = 1
                # if b_new
                Ys[k, :3] = b_new
                # print(b_new)
                k += 1

                x, y, z = [a[0, 0], b_new[0, 0]], [a[0, 1], b_new[0, 1]], [a[0, 2], b_new[0, 2]]

        self.initial_atoms_transformed_region = Xs
        self.final_atoms_transformed_region = Ys


    def write_images(self,folder, image_num):
        def write_header(f,natoms,box):
            eps = 0.1
            f.write("#LAMMPS data file\n")
            f.write("%d atoms\n" % (natoms))
            f.write("2 atom types\n")
            f.write("%0.10f %0.10f xlo xhi\n" % (box[0, 0] - eps, box[0, 1] + eps))
            f.write("%0.10f %0.10f ylo yhi\n" % (box[1, 0], box[1, 1]))
            f.write("%0.10f %0.10f zlo zhi\n" % (box[2, 0], box[2, 1]))
            f.write("0.0 0.0 0.0 xy xz yz\n\n")
            f.write("Atoms # atomic\n\n")
        def write_atoms(f,atoms):
            for i in range(len(atoms)):
                f.write("%d %d %f %f %f\n" % (int(atoms[i, 0]), int(atoms[i, 1]), atoms[i, 2], atoms[i, 3],atoms[i, 4]))

        box = self.box
        sigma = self.sigma
        inc = self.inclination
        elem = self.element
        # Write image 0 (flat gb)
        if image_num == 1:
            initial = self.initial_grain
            file = "data." + elem + "s" + str(sigma) + "inc" + str(inc) + "__step0"
            with open(folder + file, "w") as f0:
                write_header(f0,len(initial),box)
                write_atoms(f0,initial)
            print(f"Done writing bicrystal {folder}/{file}")

        final = self.final_grain
        Ys = self.final_atoms_transformed_region
        initial = self.initial_grain
        Xs = self.initial_atoms_transformed_region
        if self.step_height >= 0:
            grain_num = 2
        else:
            grain_num = 1

        # Build lookup dictionaries for quick ID-based matching
        Xs_dict = {int(x[3]): x for x in Xs}
        final = final[final[:, 0].argsort()]
        final_dict = {int(f[0]): f for f in final}
        #print(Xs_dict,Ys)
        # Populate a matrix with correct atom indices and positions
        final_atoms = []
        for i in range(len(initial)):
            atom_id = int(initial[i,0])
            if atom_id in Xs_dict:
                j = int(Xs_dict[atom_id][4])
                final_atoms.append(np.array([atom_id, grain_num, Ys[j, 0], Ys[j, 1], Ys[j, 2]]))
            elif atom_id in final_dict:
                index = int(final_dict[atom_id][0])-1
                final_atoms.append(final[index,:])

        final_atoms = np.array(final_atoms)


        # Write data
        file = "data." + elem + "s" + str(sigma) + "inc" + str(inc) + "__step"+str(image_num)
        with open(folder + file, "w") as f:
            write_header(f,len(final_atoms),box)
            write_atoms(f,final_atoms)
        print(f"Done writing bicrystal {folder}/{file}")



    def _write_neb_input_file(self,folder, image_num):
        sigma = self.sigma
        file = "data.Cus" + str(sigma) + "inc0.0__step" + str(image_num)
        outfile = "data.Cus" + str(sigma) + "inc0.0_out_step" + str(image_num)
        in_file = folder + file
        out_file = folder + outfile
        data = read_LAMMPS_datafile(in_file, 1)
        natoms = data[0][0]
        atoms = data[0][3]
        f = open(out_file, "w")
        f.write("%d\n" % (natoms))
        for i in range(len(atoms)):
            f.write("%d %f %f %f\n" % (atoms[i, 0], atoms[i, 2], atoms[i, 3], atoms[i, 4]))
        f.close()
        print("Done writing bicrystal " + out_file)

    @staticmethod
    def _threshold(g, cutoff):
        G = abs(g)
        row_ind, col_ind = np.where(g < cutoff)
        for i in range(len(row_ind)):
            G[row_ind[i], col_ind[i]] = 0
        return G

    @staticmethod
    def plot_min_shuffle_3d(Xs, Ys, box,sigma,reg_param):
        # Plot simulation box
        fig, ax = plt.subplots()
        xlo = box[0, 0]
        ylo = box[1, 0]
        zlo = box[2, 0]
        xhi = box[0, 1]
        yhi = box[1, 1]
        zhi = box[2, 1]
        ax.plot([xlo, xhi], [yhi, yhi], [zhi, zhi], 'k', linewidth=1)
        ax.plot([xlo, xhi], [ylo, ylo], [zhi, zhi], 'k', linewidth=1)
        ax.plot([xlo, xhi], [yhi, yhi], [zlo, zlo], 'k', linewidth=1)
        ax.plot([xlo, xhi], [ylo, ylo], [zlo, zlo], 'k', linewidth=1)
        ax.plot([xlo, xlo], [ylo, yhi], [zhi, zhi], 'k', linewidth=1)
        ax.plot([xlo, xlo], [ylo, yhi], [zlo, zlo], 'k', linewidth=1)
        ax.plot([xhi, xhi], [ylo, yhi], [zlo, zlo], 'k', linewidth=1)
        ax.plot([xhi, xhi], [ylo, yhi], [zhi, zhi], 'k', linewidth=1)
        ax.plot([xlo, xlo], [ylo, ylo], [zlo, zhi], 'k', linewidth=1)
        ax.plot([xlo, xlo], [yhi, yhi], [zlo, zhi], 'k', linewidth=1)
        ax.plot([xhi, xhi], [ylo, ylo], [zlo, zhi], 'k', linewidth=1)
        ax.plot([xhi, xhi], [yhi, yhi], [zlo, zhi], 'k', linewidth=1)

        # Plot particles
        ax.scatter(Xs[:, 0], Xs[:, 1], Xs[:, 2], color='red', s=2, label="Initial")
        ax.scatter(Ys[:, 0], Ys[:, 1], Ys[:, 2], color='blue', s=2, label="Final")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        name = "sigma" + str(sigma) + "inc0.0_optimalBeta_min_shuffle_eps" + str(reg_param) + ".png"
        return 1

    @staticmethod
    def plot_min_shuffle_2d(folder,Xbasis,Ybasis,Gamma,sigma,reg_param):
        plt.figure(dpi=200)
        ot.plot.plot2D_samples_mat(Xbasis, Ybasis, Gamma, c=[.5, .5, 1])
        plt.scatter(Xbasis[:, 0], Xbasis[:, 1], color='b', label='Source samples')
        plt.scatter(Ybasis[:, 0], Ybasis[:, 1], color='r', label='Target samples')
        plt.legend(loc='best')
        name = "Sigma" + str(sigma) + "min shuffle optimal beta eps=" + str(reg_param)
        plt.savefig(folder+name)