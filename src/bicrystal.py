import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from src.bicrystallography import *
from src.plastic_slip import dislocation_dipole

class bicrystal:
    """
    Represents a bicrystal
    """
    def __init__(self,gb_data,axis,lat_par,lat_Vec,
                 size_along_period=1,size_along_tilt_axis=1,
                 non_periodic_direction_size=50):
        """
        Initializes a bicrystal configuration based on grain boundary crystallography.

        Args:
            gb_data (np.ndarray): Grain boundary bicrystallographic data, typically
                the output from a bicrystallography class.
            axis (np.ndarray): Tilt axis of the bicrystal.
            lat_par (float): Lattice parameter of the material.
            lat_Vec (np.ndarray): Lattice vectors of the material.
            size_along_period (int, optional): Number of CSL repeats along the GB period.
                Defaults to 1.
            size_along_tilt_axis (int, optional): Number of CSL repeats along the tilt axis.
                Defaults to 1.
            non_periodic_direction_size (float, optional): Size of the bicrystal normal
                to the GB plane, in angstroms. Defaults to 50.

        Attributes:
            gb_data (np.ndarray): Grain boundary data input.
            axis (np.ndarray): Tilt axis.
            lat_par (float): Lattice parameter.
            lat_Vec (np.ndarray): Lattice vectors.
            size_along_period (int): Size along GB period (CSL multiples).
            size_along_tilt_axis (int): Size along tilt axis (CSL multiples).
            non_periodic_direction_size (float): Size normal to the GB plane in angstroms.
            nCells (int): Number of unit cells in the bicrystal. Computed as
                `size_factor * size_along_period`, where `size_factor` is
                40 if `size_along_period == 1`, else 25.

            grain1_orientation (Any): Orientation matrix or parameters for grain 1 (initialized as None).
            grain2_orientation (Any): Orientation matrix or parameters for grain 2 (initialized as None).
            grain1_flatgb (Any): Flat grain boundary structure for grain 1 (initialized as None).
            grain2_flatgb (Any): Flat grain boundary structure for grain 2 (initialized as None).
            grain1 (Any): Final structure of grain 1 (initialized as None).
            grain2 (Any): Final structure of grain 2 (initialized as None).
            box (Any): Simulation or bounding box information (initialized as None).
        """
        self.gb_data = gb_data
        self.axis = axis
        self.lat_par = lat_par
        self.lat_Vec = lat_Vec
        self.size_along_period = size_along_period
        self.size_along_tilt_axis = size_along_tilt_axis
        self.non_periodic_direction_size = non_periodic_direction_size
        size_factor = 40 if size_along_period==1 else 25
        self.nCells = size_factor*size_along_period

        self.grain1_orientation = None
        self.grain2_orientation = None
        self.grain1_flatgb = None
        self.grain2_flatgb = None
        self.grain1 = None
        self.grain2 = None
        self.box = None

    def _setup_bicrystal(self):
        """
            Computes the orientation matrices for the two grains in a bicrystal system,
            based on the tilt axis, misorientation, and inclination angle from the grain
            boundary (GB) data.

            This method performs:
                - Construction of grain rotation matrices using Rodrigues’ rotation formula.
                - Application of inclination and misorientation angles.
                - Alignment of the tilt axis with the [001] direction via coordinate transformation.
                - Final transformation of lattice vectors into crystal orientations.

            Sets:
                grain1_orientation (np.ndarray): Rotated lattice vectors for grain 1 in the final coordinate system.
                grain2_orientation (np.ndarray): Rotated lattice vectors for grain 2 in the final coordinate system.

            Notes:
                - The tilt axis is aligned to the global Z-axis [0, 0, 1].
                - Uses Rodrigues’ formula to compute rotations from GB data.
                - Applies an additional rotation if the tilt axis is [1 1 1] or [1 1 0], to align the grains properly.
        """
        dim = 3
        axis    = self.axis
        lat_par = self.lat_par
        gb_data = self.gb_data
        lat_Vec = self.lat_Vec

        # Rotate basis vectors to get A and B
        axis_true = axis
        axis = axis / la.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        misorientation = gb_data[1]
        theta1 = 1 * (misorientation / 2) * np.pi / 180
        theta2 = 1 * (misorientation / 2) * np.pi / 180
        inclination = gb_data[2]
        phi = (inclination) * np.pi / 180
        grain1_rotation = np.eye(3) + np.sin(theta1 + phi) * K + (1 - np.cos(theta1 + phi)) * np.matmul(K, K)
        grain2_rotation = np.eye(3) + np.sin(-theta2 + phi) * K + (1 - np.cos(-theta2 + phi)) * np.matmul(K, K)
        g1 = np.matmul(grain1_rotation, lat_Vec)
        g2 = np.matmul(grain2_rotation, lat_Vec)

        # Rotate crystals such that tilt axis lies along [001]
        z = np.array([0, 0, 1])
        ax = np.cross(axis, z)
        if axis[0] == z[0] and axis[1] == z[1] and axis[2] == z[2]:
            final_rotation = np.eye(dim)
        else:
            ax = ax / la.norm(ax)
            V = np.array([[0, -ax[2], ax[1]],
                          [ax[2], 0, -ax[0]],
                          [-ax[1], ax[0], 0]])
            rotation_angle = -np.arccos(np.dot(axis, z) / la.norm(axis))
            rotation = np.eye(dim) + np.sin(rotation_angle) * V + np.matmul(V, V) * (1 - np.cos(rotation_angle))

            if axis_true[0] == 1 and axis_true[1] == 1 and axis_true[2] == 1:
                rotation_tilt_axis = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
                                   [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
                                   [0, 0, 1]])
            if axis_true[0] == 1 and axis_true[1] == 1 and axis_true[2] == 0:
                rotation_tilt_axis = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
                                   [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
                                   [0, 0, 1]])
            final_rotation = np.matmul(rotation, rotation_tilt_axis)

        grain1_orientation = np.matmul(final_rotation.T, g1)
        grain2_orientation = np.matmul(final_rotation.T, g2)
        self.grain1_orientation = grain1_orientation
        self.grain2_orientation = grain2_orientation

    def create_flat_gb_bicrystal(self,gb_position):
        """
            Generates the initial flat grain boundary (GB) structure by constructing a dichromatic pattern
            from the two grain orientations and selecting atoms to form a bicrystal.

            This function:
                - Constructs a 3D dichromatic pattern for both grains using lattice translations.
                - Filters atoms within a simulation box, optionally shifted.
                - Selects atoms for each grain based on their position relative to the GB plane.
                - Removes duplicate atoms using a distance-based uniqueness check.

            Args:
                gb_position (float): The X-coordinate (along GB normal) at which the GB plane is located.

            Sets:
                grain1_flatgb (np.ndarray): Filtered atomic positions for grain 1, including unique atom IDs.
                grain2_flatgb (np.ndarray): Filtered atomic positions for grain 2, including unique atom IDs.
                box (np.ndarray): Dimensions of the simulation box used for filtering and placement.

            Notes:
                - Atom uniqueness is checked using `_check_unique()` with a distance tolerance of 0.5 Å.
                - Atoms on or near the GB plane are discarded to create a clean flat interface.
                - The structure is built from `-nCells` to `+nCells` in lattice units in all three dimensions.
        """
        print("\n============================= Generating initial flat GB  =======================================")
        # Declare relevant variables
        axis = self.axis
        lat_par = self.lat_par
        gb_data = self.gb_data
        lat_Vec = self.lat_Vec
        period = gb_data[3]*lat_par
        nCells = self.nCells

        grainA = self.grain1_orientation
        grainB = self.grain2_orientation
        grainA_dichromatic = []
        grainB_dichromatic = []

        box = np.array([[self.size_along_period * period , self.non_periodic_direction_size, la.norm(axis) * lat_par * self.size_along_tilt_axis],
                        [self.non_periodic_direction_size, self.size_along_period * period , la.norm(axis) * lat_par * self.size_along_tilt_axis]])
        eps_p = np.array([0,-0.1,0.1])
        eps_n = np.array([0,0.0,0.0])
        box_shift = np.array([0,0,0])

        # Create dichromatic pattern
        for nx in range(-nCells, nCells):
            for ny in range(-nCells, nCells):
                for nz in range(-nCells, nCells):
                    loc = np.array([[nx], [ny], [nz]])
                    a = lat_par * np.matmul(grainA, loc)[:, 0]
                    if np.all((a-box_shift>-box[1,:]-eps_n) & (a-box_shift<box[1,:]+eps_p)):
                        grainA_dichromatic.append([a[0], a[1] , a[2]])
                    b = lat_par * np.matmul(grainB, loc)[:, 0]
                    if np.all((b-box_shift>-box[1,:]-eps_n) & (b-box_shift<box[1,:]+eps_p)):
                        grainB_dichromatic.append([b[0], b[1] , b[2]])


        # Creation of a bicrystal by deleting atoms on the GB
        gA = []
        gB = []
        eps = 0.1
        atom_count = 1
        for i in range(len(grainA_dichromatic)):
            if grainA_dichromatic[i][0] >= gb_position - eps:
                point = [grainA_dichromatic[i][0], grainA_dichromatic[i][1], grainA_dichromatic[i][2]]
                flag = bicrystal._check_unique(point, gA, 0.5)
                if flag == 1:
                    point.append(atom_count)
                    gA.append(point)
                    atom_count += 1

        for i in range(len(grainB_dichromatic)):
            if grainB_dichromatic[i][0] < gb_position - eps:
                point = [grainB_dichromatic[i][0], grainB_dichromatic[i][1], grainB_dichromatic[i][2]]
                flag = bicrystal._check_unique(point, gB, 0.5)
                if flag == 1:
                    point.append(atom_count)
                    gB.append(point)
                    atom_count += 1

        self.grain1_flatgb = np.array(gA)
        self.grain2_flatgb = np.array(gB)
        self.box = box


    def create_disconnection_containing_bicrystal(self,nodes,burgers_vector,step_height,gb_position,
                                                  image_number,nImages = 2,number_of_dipoles=3):
        """
            Generates a grain boundary (GB) image with a disconnection dipole and possible steps
            based on the bicrystallographic configuration.

            This method:
                - Validates the presence of a flat GB structure.
                - Determines regions affected by the dislocation dipole.
                - Applies displacement fields to atoms in the transformed region based on
                  the dipole configuration and Burgers vector.
                - Combines atoms from transformed and non-transformed regions to form a new GB image.
                - Assigns unique atom IDs while avoiding duplicate atoms.

            Args:
                image_number (int): Identifier for the GB image being generated, used to determine
                    whether dipole transitions are included.
                gb_position (float): X-coordinate of the GB plane, defining the boundary between grains.
                nodes (np.ndarray): 2x2 array of disconnection node coordinates in the GB plane.
                step_height (float): Height of the step at the GB (positive or negative), used to
                    determine which grain is displaced.
                burgers_vector (np.ndarray): Displacement vector representing the dislocation.
                number_of_dipoles (int): Number of dipoles introduced in the bicrystal.

            Sets:
                grain1 (np.ndarray): Atom positions for grain 1, including plastic displacement and unique IDs.
                grain2 (np.ndarray): Atom positions for grain 2, similarly displaced and labeled.

            Raises:
                ValueError: If the flat GB structure has not been created via `create_flat_gb_bicrystal`.

            Notes:
                - Atoms in the transformed region are displaced based on the disconnection geometry.
                - Overlapping atoms are checked using Euclidean distance and matched to maintain continuity.
                - Diagnostics are written to file if mismatch occurs between atom counts in transformed regions.

            Diagnostic Output:
                If atom mismatches are detected in the transformed region, a diagnostic file is written to:
                `/Users/hj-home/Desktop/Research/GB_kinetics_oop/output/grain_diagnostics.txt`
        """

        print("\n================= Generating GB image " + str(image_number) + " bicrystallographically ==============")
        # Declare relevant variables
        axis = self.axis
        lat_par = self.lat_par
        gb_data = self.gb_data
        period = gb_data[3] * lat_par
        nCells = self.nCells


        if self.grain1_flatgb is None or self.grain2_flatgb is None:
            raise ValueError("Bicrystal with flat GB not constructed. Run create_flat_gb_bicrystal first.")
        grain1_flat = self.grain1_flatgb
        grain2_flat = self.grain2_flatgb
        box = self.box

        # Decompose dislocation dipole in case of a stepped boundary
        tol = 0.0
        disconnection_start = nodes[0, 0] - tol
        disconnection_stop  = nodes[1, 0] + tol
        diag_plt = False  # True
        nodes_for_solid_angle = np.zeros((2, 2))
        if image_number < 2 * self.size_along_period:
            nodes_modified = [nodes]  # + np.array([[-period/8,0],[period/8,0]])
            if number_of_dipoles > 1:
                transition_node_start = np.array([[nodes[0, 0], gb_position], [nodes[0, 0], nodes[0, 1]]])
                transition_node_stop = np.array([[nodes[1, 0], nodes[1, 1]], [nodes[1, 0], gb_position]])
                nodes_modified.append(transition_node_start)
                nodes_modified.append(transition_node_stop)
        else:
            nodes_modified = [nodes + np.array([[-period / 4, 0], [period / 4, 0]])]
            dipole_number = 1

        # Create atoms in the transformed region with displacement due to dislocation dipole
        transformed_atoms = []
        box_shift = np.array([0, 0, 0])
        grainA = self.grain1_orientation
        grainB = self.grain2_orientation

        disconnection_start = disconnection_start if disconnection_start > -box[1, 1] else -box[1, 1]
        disconnection_stop  = disconnection_stop  if disconnection_stop  <  box[1, 1] else  box[1, 1]
        if step_height < 0:
            grain2transform = grainA
            lower_bounds = np.array([gb_position + step_height-0.1, disconnection_start, -box[1, 2]])
            upper_bounds = np.array([gb_position-0.1, disconnection_stop, box[1, 2]])
            eps_n = np.array([0, 0, 0.0])
            eps_p = np.array([0, -0.1, 0.1])
        else:
            grain2transform = grainB
            lower_bounds = np.array([gb_position-0.1, disconnection_start, -box[1, 2]])
            upper_bounds = np.array([gb_position + step_height, disconnection_stop, box[1, 2]])
            eps_n = np.array([0, 0, 0.0])
            eps_p = np.array([0, -0.1, 0.1])

        for nx in range(-nCells, nCells):
            for ny in range(-nCells, nCells):
                for nz in range(-nCells, nCells):
                    loc = np.array([[nx], [ny], [nz]])
                    a = lat_par * np.matmul(grain2transform, loc)[:, 0]
                    if np.all((a - box_shift > lower_bounds - eps_n) & (a - box_shift < upper_bounds + eps_p)):
                        point = np.array([a[1], a[0]])
                        plastic_displacement = bicrystal._apply_plastic_displacement(nodes_modified,period,burgers_vector,point,box[1,1],-box[1,1])
                        if a[1]-plastic_displacement > disconnection_stop:
                            plastic_displacement +=(disconnection_stop - disconnection_start)
                        elif a[1]-plastic_displacement < disconnection_start:
                            plastic_displacement -= (disconnection_start - disconnection_start)
                        transformed_atoms.append([a[0], a[1] - plastic_displacement, a[2]])

        # Compile atoms in regions
        epsx = -0.1
        grain1_disconnection = []
        grain2_disconnection = []
        transformed_atoms_initial = []
        # Fill up atoms in the non-transformed region
        if step_height<0:
            for atoms in grain2_flat:
                if(atoms[0] <gb_position+step_height +epsx or
                        ((atoms[1]<disconnection_start or atoms[1]>disconnection_stop) and atoms[0]<gb_position+epsx)):
                    point = np.array([atoms[1],atoms[0]])
                    plastic_displacement = bicrystal._apply_plastic_displacement(nodes_modified,period,burgers_vector,point,box[1,1],-box[1,1])
                    grain2_disconnection.append([atoms[0],atoms[1] - plastic_displacement, atoms[2],atoms[3]])
                else:
                    transformed_atoms_initial.append([atoms[0],atoms[1], atoms[2],atoms[3]])

            for atoms in grain1_flat:
                if atoms[0]>gb_position+epsx:
                    point = np.array([atoms[1],atoms[0]])
                    plastic_displacement = bicrystal._apply_plastic_displacement(nodes_modified, period, burgers_vector, point,box[1, 1],-box[1,1])
                    grain1_disconnection.append([atoms[0], atoms[1] - plastic_displacement, atoms[2], atoms[3]])
        else:
            for atoms in grain1_flat:
                if (atoms[0] > gb_position + step_height + epsx or
                        ((atoms[1] < disconnection_start or atoms[1] > disconnection_stop) and atoms[0] > gb_position + epsx)):
                    point = np.array([atoms[1], atoms[0]])
                    plastic_displacement = bicrystal._apply_plastic_displacement(nodes_modified, period, burgers_vector, point,box[1, 1],-box[1,1])
                    grain1_disconnection.append([atoms[0], atoms[1] - plastic_displacement, atoms[2], atoms[3]])
                else:
                    transformed_atoms_initial.append([atoms[0], atoms[1], atoms[2], atoms[3]])

            for atoms in grain2_flat:
                if atoms[0] < gb_position - epsx:
                    point = np.array([atoms[1], atoms[0]])
                    plastic_displacement = bicrystal._apply_plastic_displacement(nodes_modified, period, burgers_vector, point,box[1, 1],-box[1,1])
                    grain2_disconnection.append([atoms[0], atoms[1] - plastic_displacement, atoms[2], atoms[3]])

        # Populate the transformed region
        if len(transformed_atoms_initial) != len(transformed_atoms):
            print("Atomic construction failed! Transformed region is not defined well")
            print("Atoms in transformed region in base bicrystal:"+ str(len(transformed_atoms_initial)))
            print("Atoms in transformed region in new bicrystal:"+ str(len(transformed_atoms)))
            filename =  "/Users/hj-home/Desktop/Research/GB_kinetics_oop/output/grain_diagnostics.txt"
            self._diagnostic_grain_writing(np.array(transformed_atoms_initial), np.array(transformed_atoms), filename)
        for atom in transformed_atoms:
            min_dist = 1e5
            if len(transformed_atoms_initial) > 0:
                for j in range(len(transformed_atoms_initial)):
                    b_p = np.array(atom)
                    a_p = transformed_atoms_initial[j]
                    dist = la.norm(b_p - a_p[:3])
                    if min_dist > dist:
                        atom_count = a_p[3]
                        index = j
                        min_dist = dist
                transformed_atoms_initial.pop(index)
                atom.append(atom_count)
                if step_height<0:
                    grain1_disconnection.append(atom)
                else:
                    grain2_disconnection.append(atom)

        self.grain1 = np.array(grain1_disconnection)
        self.grain2 = np.array(grain2_disconnection)

    def create_fix_eco_orientationfile(self,folder):
        """
            Writes the grain orientations to a `.ori` file compatible with the fix_eco
            command in LAMMPS, saving it to the specified folder.

            The file contains the lattice orientations of both grains, scaled by the
            lattice parameter, and is named based on the grain boundary properties.

            Args:
                folder (str): Directory path where the orientation file will be saved.
                    The folder path should end with a slash ('/').

            Raises:
                ValueError: If the bicrystal has not been fully set up (i.e., orientations
                    are None). You must run `_setup_bicrystal` before calling this method.

            Outputs:
                Creates a file named `Sigma{sigma}_mis{mis}_inc{inc}.ori` in the given folder,
                where `sigma`, `mis` (misorientation), and `inc` (inclination) are from `gb_data`.
                The file contains the orientation matrices for the two grains, row-wise.

            Prints:
                Confirmation message indicating the successful creation of the orientation file.
        """
        if self.grain1_orientation is None or self.grain2_orientation is None:
            raise ValueError("No bicrystal fully setup. Run _setup_bicrystal first.")
        first_grain = self.grain1_orientation.T * self.lat_par
        second_grain = self.grain2_orientation.T * self.lat_par
        gb_props = self.gb_data
        sigma = gb_props[0]
        mis = gb_props[1]
        inc = gb_props[2]
        file = "Sigma" + str(sigma) + "_mis" + str(mis) + "_inc" + str(inc) + ".ori"
        f = open(folder + file, "w")
        for i in range(first_grain.shape[0]):
            f.write("%f %f %f\n" % (first_grain[i, 0], first_grain[i, 1], first_grain[i, 2]))
        for i in range(second_grain.shape[0]):
            f.write("%f %f %f\n" % (second_grain[i, 0], second_grain[i, 1], second_grain[i, 2]))
        f.close()
        print("Done writing fix eco orientation file : " + folder + file)

    def write(self,folder, elem, suffix,mode=2):
        """
            Writes the bicrystal atomic configuration to a LAMMPS data file.

            The output file contains atom positions for both grains along with box
            dimensions and metadata required for LAMMPS simulations.

            Args:
                folder (str): Directory path where the data file will be saved.
                              Should end with a '/' or use os.path.join for safety.
                elem (str): Element symbol or identifier for naming the output file.
                suffix (str): Suffix string appended to the output filename.
                mode (int, optional): Index selecting the dimension for box size from
                                      self.box (default is 2).

            Returns:
                str: The filename of the written data file.

            Raises:
                AttributeError: If bicrystal data (e.g., grain arrays or box) is not set.

            Outputs:
                Creates a LAMMPS data file named as:
                `data.{elem}s{sigma}inc{inc}_{suffix}`, containing atomic coordinates
                and simulation box details.

            Prints:
                Confirmation message after successful file write.
        """
        gb_data = self.gb_data
        sigma = gb_data[0]
        mis = gb_data[1]
        inc = gb_data[2]
        g_A = self.grain1
        g_B = self.grain2
        box = self.box

        file = "data." + elem + "s" + str(sigma) + "inc" + str(inc) + "_" + suffix
        name = folder + file
        natoms = g_A.shape[0] + g_B.shape[0]
        f = open(name, "w")
        eps = 0.1
        f.write("# LAMMPS data file Sigma = %d, inclination = %f\n"%(sigma,inc))
        f.write("#LAMMPS data file\n")
        f.write("%d atoms\n" % (natoms))
        f.write("2 atom types\n")
        f.write("%0.10f %0.10f xlo xhi\n" % (-box[mode - 1, 0] - eps, box[mode - 1, 0] + eps))
        f.write("%0.10f %0.10f ylo yhi\n" % (-box[mode - 1, 1], box[mode - 1, 1]))
        f.write("%0.10f %0.10f zlo zhi\n" % (-box[mode - 1, 2], box[mode - 1, 2]))
        box = np.array([[-box[mode - 1, 0] - eps, box[mode - 1, 0] - eps], [-box[mode - 1, 1], box[mode - 1, 1]],
                        [-box[mode - 1, 2], box[mode - 1, 2]]])
        f.write("0.0 0.0 0.0 xy xz yz\n\n")
        f.write("Atoms # atomic\n\n")
        k = 1
        grain_A = []
        grain_B = []
        for i in range(g_A.shape[0]):
            grain_num = 1
            f.write("%d %d %0.10f %0.10f %0.10f\n" % (k, grain_num, g_A[i, 0], g_A[i, 1], g_A[i, 2]))
            grain_A.append([g_A[i, 0], g_A[i, 1], g_A[i, 2], k])
            k += 1
        for i in range(g_B.shape[0]):
            grain_num = 2
            f.write("%d %d %0.10f %0.10f %0.10f\n" % (k, grain_num, g_B[i, 0], g_B[i, 1], g_B[i, 2]))
            grain_B.append([g_B[i, 0], g_B[i, 1], g_B[i, 2], k])
            k += 1
        f.close()

        print("Done writing bicrystal " + name)
        return file

    def ordered_write(self,folder, elem, image_num, xpos, h=0,start=0,stop = 0,mode=2):
        """
             Write a LAMMPS data file of the bicrystal atomic configuration including dislocation or disconnection region labeling.

        Args:
            folder (str): Directory path to save the output file.
            elem (str): Element symbol or identifier used in filename.
            h (int): Flag indicating which grain configuration to use:
                     0 for flat grain boundary, else dislocated configuration.
            image_num (int): Identifier for the image number in the filename.
            xpos (float): Position value used for grain classification.
            start (float): Lower bound along y-axis for dislocation region.
            stop (float): Upper bound along y-axis for dislocation region.
            mode (int, optional): Index to select box dimension (default 2).

        Returns:
            str: The filename of the written data file.

        Outputs:
            Writes a LAMMPS data file named like
            'data.{elem}s{sigma}inc{inc}_size{size}disc{image_num}'.
        """
        gb_data = self.gb_data
        sigma = gb_data[0]
        mis = gb_data[1]
        inc = gb_data[2]
        if h == 0:
            g_A = self.grain1_flatgb
            g_B = self.grain2_flatgb
        else:
            g_A = self.grain1
            g_B = self.grain2
        box = self.box
        size = self.size_along_period
        file = "data." + elem + "s" + str(sigma) + "inc" + str(inc) + "_size" +str(size)+"disc"+str(image_num)
        name = folder + file
        natoms = g_A.shape[0] + g_B.shape[0]
        f = open(name, "w")
        eps = 0.1
        f.write("# LAMMPS data file Sigma = %d, inclination = %f\n"%(sigma,inc))
        f.write("#LAMMPS data file\n")
        f.write("%d atoms\n" % (natoms))
        f.write("2 atom types\n")
        f.write("%0.10f %0.10f xlo xhi\n" % (-box[mode - 1, 0] - eps, box[mode - 1, 0] + eps))
        f.write("%0.10f %0.10f ylo yhi\n" % (-box[mode - 1, 1], box[mode - 1, 1]))
        f.write("%0.10f %0.10f zlo zhi\n" % (-box[mode - 1, 2], box[mode - 1, 2]))
        box = np.array([[-box[mode - 1, 0] - eps, box[mode - 1, 0] - eps], [-box[mode - 1, 1], box[mode - 1, 1]],
                        [-box[mode - 1, 2], box[mode - 1, 2]]])
        f.write("0.0 0.0 0.0 xy xz yz\n\n")
        f.write("Atoms # atomic\n\n")
        k = 1
        grain_A = []
        grain_B = []
        bicrystal = []
        for i in range(g_A.shape[0]):
            grain_num = 1
            row = np.array([g_A[i, 0], g_A[i, 1], g_A[i, 2], g_A[i, 3], grain_num])
            bicrystal.append(row)
        for i in range(g_B.shape[0]):
            if abs(g_B[i, 0] - xpos - h) < 0.01 and g_B[i, 1] > start and g_B[i, 1] < stop:
                grain_num = 1
            else:
                grain_num = 2
            row = np.array([g_B[i, 0], g_B[i, 1], g_B[i, 2], g_B[i, 3], grain_num])
            bicrystal.append(row)
        b = np.array(bicrystal)
        b = b[b[:, 3].argsort()]
        for i in range(b.shape[0]):
            f.write("%d %d %0.10f %0.10f %0.10f\n" % (b[i, 3], b[i, 4], b[i, 0], b[i, 1], b[i, 2]))
        f.close()
        print("Done writing bicrystal " + name)
        return file

    @staticmethod
    def _apply_plastic_displacement(nodes,period,b,point,boxlimhi,boxlimlo):
        """
            Calculate the total plastic displacement at a given point due to a set of dislocation dipoles.

            Args:
                nodes (list or array): List of node pairs defining dislocation dipoles.
                period (float): Periodicity length along the boundary.
                b (float): Burgers vector of the dislocation.
                point (array-like): Coordinates [x, y] at which displacement is calculated.
                boxlimhi (float): Upper boundary limit along the gb period.
                boxlimlo (float): Lower boundary limit along the gb period.

            Returns:
                float: Total plastic displacement at the point accounting for periodic boundary conditions.
        """
        displacement = 0
        for dislocation_nodes in nodes:
            dipole = dislocation_dipole(dislocation_nodes, period, b)
            solidAngle, disp_temp = dipole.plastic_displacement(point)
            displacement += disp_temp
            del dipole
        box_length = boxlimhi-boxlimlo
        if point[0] - displacement > boxlimhi:
            displacement += box_length
        elif point[0] - displacement < boxlimlo:
            displacement -= box_length
        return displacement

    @staticmethod
    def _check_unique(point,grain,cutoff):
        """
            Check if a point is unique in the grain within a given cutoff distance.

            Args:
                point (list or array-like): Coordinates [x, y, z] of the point to check.
                grain (list of lists or arrays): Collection of points representing the grain.
                cutoff (float): Distance threshold to determine uniqueness.

            Returns:
                int: 1 if point is unique, 0 if a similar point exists within cutoff.
        """
        for i in range(len(grain)):
            if (abs(grain[i][0] - point[0]) < cutoff
            and abs(grain[i][1] - point[1]) < cutoff
            and abs(grain[i][2] - point[2]) < cutoff):
                return 0
        return 1

    @staticmethod
    def _diagnostic_grain_writing(grain1,grain2,filename):
        """
            Write diagnostic grain data to a file.

            Args:
                grain1 (np.ndarray): Array of grain1 atoms with at least 3 columns (x,y,z).
                grain2 (np.ndarray): Array of grain2 atoms with at least 3 columns (x,y,z).
                filename (str): Path to the output file.
        """
        total_atoms = len(grain1) + len(grain2)
        with open(filename, "w") as f:
            f.write(f"{total_atoms}\n\n")
            for atom in grain1:
                f.write(f"{atom[0]:.6f} {atom[1]:.6f} {atom[2]:.6f} 1\n")
            for atom in grain2:
                f.write(f"{atom[0]:.6f} {atom[1]:.6f} {atom[2]:.6f} 2\n")

    @staticmethod
    def _diagnostic_plotting(grain1,grain2,minx,maxx,miny,maxy):
        """
            Plot two grain datasets for diagnostic visualization.

            Args:
                grain1 (array-like): Coordinates of grain 1 atoms (Nx3 or similar).
                grain2 (array-like): Coordinates of grain 2 atoms (Mx3 or similar).
                minx (float): Minimum x-axis limit.
                maxx (float): Maximum x-axis limit.
                miny (float): Minimum y-axis limit.
                maxy (float): Maximum y-axis limit.
        """
        g1 = np.array(grain1)
        g2 = np.array(grain2)

        plt.figure(dpi=200, figsize=(3, 3))
        plt.scatter(g1[:, 1], g1[:, 0], s=1, color="red", label="Grain 1")
        plt.scatter(g2[:, 1], g2[:, 0], s=1, color="blue", label="Grain 2")
        plt.xlim(minx, maxx)
        plt.ylim(miny, maxy)
        plt.xlabel("Y coordinate")
        plt.ylabel("X coordinate")
        plt.legend()
        plt.title("Grain Boundary Diagnostic Plot")
        plt.tight_layout()
        plt.show()
