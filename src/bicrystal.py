import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from scipy.spatial import cKDTree
from src.bicrystallography import *
from src.plastic_slip import dislocation_dipole

class bicrystal:
    """
    Represents a bicrystal
    """

    # Tie-breaker for atoms sitting exactly on a periodic box face.  It has to be
    # far larger than the round-off of the rotated coordinates (~1e-13 A) and far
    # smaller than any interplanar spacing, otherwise a whole plane of atoms drops
    # in or out of the cell depending on the last bit of the rotation matrix.
    # Every periodic face is half-open: [-L, +L).
    _EPS_FACE = 1e-6
    # Tie-breaker for the GB / cut plane, which passes exactly through a plane of
    # atoms.  The coincident plane is assigned to grain A.
    _EPS_GB = 1e-4
    # Two lattice sites closer than this are the same site generated twice.  The
    # shortest interatomic distance in any real lattice is ~1e3 times larger, so
    # this only ever removes genuine duplicates - unlike a cutoff of the order of
    # the lattice parameter, which silently deletes distinct atoms.
    _EPS_DUPLICATE = 1e-3
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

        self.grain1_orientation = None
        self.grain2_orientation = None
        self.grain1_flatgb = None
        self.grain2_flatgb = None
        self.grain1 = None
        self.grain2 = None
        self.box = None

        # Reference (undisplaced) dichromatic pattern, built once and shared by the
        # flat GB and every disconnection image.  See _build_reference_lattice.
        self._refA = None
        self._refB = None
        # Boolean masks and atom IDs of the flat GB, indexed into _refA / _refB.
        self._maskA_flat = None
        self._maskB_flat = None
        self._idA = None
        self._idB = None

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

    def _build_reference_lattice(self):
        """
            Generates the reference (undisplaced) dichromatic pattern once.

            Both the flat GB and every disconnection image are selected out of these
            two arrays, so an atom that survives into both configurations is literally
            the same row of the same array.  That is what makes the atom IDs of image 0
            and image k refer to the same physical atom, which the whole downstream
            pipeline relies on: `min_shuffle.format_input` pairs the two configurations
            by atom ID alone, and `min_shuffle.run` refuses to proceed unless the two
            sets have the same size.

            Only the lattice points that can land inside the box are enumerated.  The
            required range is derived from the box corners rather than from a fixed
            index bound, which for `size_along_period = 4` enumerated 8e6 points per
            grain and discarded over 99% of them.

            Sets:
                _refA (np.ndarray): (N,3) reference sites of grain 1, deduplicated and
                    in a deterministic (z, y, x) order.
                _refB (np.ndarray): (M,3) reference sites of grain 2, likewise.
                box (np.ndarray): Simulation box (unchanged convention).
        """
        axis = self.axis
        lat_par = self.lat_par
        period = self.gb_data[3] * lat_par

        box = np.array([[self.size_along_period * period , self.non_periodic_direction_size, la.norm(axis) * lat_par * self.size_along_tilt_axis],
                        [self.non_periodic_direction_size, self.size_along_period * period , la.norm(axis) * lat_par * self.size_along_tilt_axis]])

        eps_face = bicrystal._EPS_FACE
        # x is a free surface, so both ends are closed; y and z are periodic and are
        # half-open, [-L, +L), so that the atom on the +L face is not a duplicate of
        # the one on the -L face.
        #
        # The tie-breaker has to be applied to BOTH ends of a periodic direction.  A
        # bare `>= -L` splits the plane of atoms sitting at -L down the middle: the
        # rotation leaves them at -L +/- 1e-13, so whichever way the last bit fell
        # decides whether each atom is kept.  Shifting the whole window by -eps_face
        # keeps all of the -L plane and drops all of the +L plane, deterministically.
        lower = np.array([-box[1, 0] - eps_face, -box[1, 1] - eps_face, -box[1, 2] - eps_face])
        upper = np.array([ box[1, 0] + eps_face,  box[1, 1] - eps_face,  box[1, 2] - eps_face])

        refs = []
        for orientation in (self.grain1_orientation, self.grain2_orientation):
            basis = lat_par * orientation                 # columns = lattice vectors
            # n = basis^-1 r is linear, so each component of n attains its extremes at
            # a corner of the box; the eight corners therefore bound the integer range.
            corners = np.array([[cx, cy, cz] for cx in (lower[0], upper[0])
                                             for cy in (lower[1], upper[1])
                                             for cz in (lower[2], upper[2])])
            n_needed = int(np.ceil(np.abs(la.solve(basis, corners.T)).max())) + 1
            grid = np.mgrid[-n_needed:n_needed+1, -n_needed:n_needed+1, -n_needed:n_needed+1]
            coords = grid.reshape(3, -1)
            pts = (basis @ coords).T
            mask = np.all((pts >= lower) & (pts <= upper), axis=1)
            pts = bicrystal._unique_atoms(pts[mask])
            # A deterministic order so the atom IDs do not depend on the order numpy
            # happened to produce.
            pts = pts[np.lexsort((pts[:, 0], pts[:, 1], pts[:, 2]))]
            refs.append(pts)

        self._refA, self._refB = refs
        self.box = box

    def create_flat_gb_bicrystal(self,gb_position):
        """
            Generates the initial flat grain boundary (GB) structure by selecting each
            grain out of the shared reference lattice.

            Args:
                gb_position (float): The X-coordinate (along GB normal) at which the GB plane is located.

            Sets:
                grain1_flatgb (np.ndarray): Atomic positions of grain 1 with atom IDs in column 3.
                grain2_flatgb (np.ndarray): Atomic positions of grain 2 with atom IDs in column 3.
                box (np.ndarray): Dimensions of the simulation box.

            Notes:
                - The coincident CSL plane at `gb_position` belongs to grain A, so it is
                  not duplicated into grain B.
                - Atom IDs run 1..N over grain 1 and then grain 2, matching the order
                  `ordered_write` writes them in.
        """
        print("\n============================= Generating initial flat GB  =======================================")
        if self._refA is None:
            self._build_reference_lattice()
        refA, refB = self._refA, self._refB
        eps_gb = bicrystal._EPS_GB

        self._maskA_flat = refA[:, 0] >= gb_position - eps_gb
        self._maskB_flat = refB[:, 0] <  gb_position - eps_gb

        gA = refA[self._maskA_flat]
        gB = refB[self._maskB_flat]

        # IDs live on the reference arrays so that every image can look up the ID of
        # a site without matching coordinates.
        self._idA = np.zeros(len(refA))
        self._idB = np.zeros(len(refB))
        self._idA[self._maskA_flat] = np.arange(1, len(gA) + 1)
        self._idB[self._maskB_flat] = np.arange(len(gA) + 1, len(gA) + len(gB) + 1)

        self.grain1_flatgb = np.column_stack([gA, self._idA[self._maskA_flat]])
        self.grain2_flatgb = np.column_stack([gB, self._idB[self._maskB_flat]])

    def create_disconnection_containing_bicrystal(self,nodes,burgers_vector,step_height,gb_position,
                                                  image_number,nImages = 2,number_of_dipoles=3):
        """
            Generates a GB image containing a disconnection dipole, selected out of the
            same reference lattice as the flat GB so that the atom IDs stay consistent.

            The two grains are separated by a stepped cut plane

                x_cut(y) = gb_position + step_height   inside the disconnection loop
                x_cut(y) = gb_position                 outside it

            with grain 1 at `x >= x_cut` and grain 2 at `x < x_cut`.  One formula covers
            both signs of `step_height`: a positive step moves the boundary up into
            grain 1, a negative one down into grain 2.

            Atom IDs are handled by bookkeeping on the reference lattice rather than by
            matching coordinates.  A site that is occupied by the same grain in both the
            flat GB and this image simply keeps its ID.  The sites one grain vacates
            inside the loop and the sites the other grain fills are equal in number (CSL
            geometry), so the freed IDs are handed to the filled sites through a
            canonical (z, y, x) ordering on both sides.  Which vacated site donates its
            ID to which filled site is physically irrelevant - `min_shuffle` re-solves
            the correspondence from scratch with optimal transport - so any bijection
            will do; all the pipeline requires is that image 0 and image k carry the
            same ID set.

            Args:
                nodes (np.ndarray): 2x2 array of disconnection node coordinates.
                burgers_vector (float): Magnitude of the Burgers vector (glide, along y).
                step_height (float): Height of the step at the GB (either sign).
                gb_position (float): X-coordinate of the flat GB plane.
                image_number (int): Identifier for the GB image being generated.
                nImages (int): Number of periodic images used for the solid angle.
                number_of_dipoles (int): Number of dipoles introduced in the bicrystal.

            Sets:
                grain1 (np.ndarray): Grain 1 atoms, displaced, with IDs in column 3.
                grain2 (np.ndarray): Grain 2 atoms, displaced, with IDs in column 3.

            Raises:
                ValueError: If the flat GB has not been built, or if the number of
                    vacated and filled sites differ (which would break the ID contract).
        """

        print("\n================= Generating GB image " + str(image_number) + " bicrystallographically ==============")
        lat_par = self.lat_par
        gb_data = self.gb_data
        period = gb_data[3] * lat_par

        if self.grain1_flatgb is None or self.grain2_flatgb is None:
            raise ValueError("Bicrystal with flat GB not constructed. Run create_flat_gb_bicrystal first.")
        refA, refB = self._refA, self._refB
        box = self.box
        eps_gb = bicrystal._EPS_GB
        eps_face = bicrystal._EPS_FACE

        # Decompose dislocation dipole in case of a stepped boundary
        disconnection_start = nodes[0, 0]
        disconnection_stop  = nodes[1, 0]
        if image_number < 2 * self.size_along_period:
            nodes_modified = [nodes]
            if number_of_dipoles > 1:
                transition_node_start = np.array([[nodes[0, 0], gb_position], [nodes[0, 0], nodes[0, 1]]])
                transition_node_stop = np.array([[nodes[1, 0], nodes[1, 1]], [nodes[1, 0], gb_position]])
                nodes_modified.append(transition_node_start)
                nodes_modified.append(transition_node_stop)
        else:
            nodes_modified = [nodes + np.array([[-period / 4, 0], [period / 4, 0]])]

        disconnection_start = max(disconnection_start, -box[1, 1])
        disconnection_stop  = min(disconnection_stop ,  box[1, 1])

        # --- Select both grains on the reference lattice ------------------------
        # A single half-open window, [start, stop), for every use of "inside the
        # loop".  The old code used a closed window when carrying atoms over from the
        # flat GB and a half-open one shrunk by 0.1 A when regenerating the stepped
        # region, and that inconsistency is what made the two atom counts diverge.
        def in_loop(p):
            return ((p[:, 1] >= disconnection_start - eps_face) &
                    (p[:, 1] <  disconnection_stop  - eps_face))

        def x_cut(p):
            return gb_position + np.where(in_loop(p), step_height, 0.0)

        maskA_step = refA[:, 0] >= x_cut(refA) - eps_gb
        maskB_step = refB[:, 0] <  x_cut(refB) - eps_gb

        # --- Transfer the atom IDs of the sites that changed grain --------------
        lost_A = np.where(self._maskA_flat & ~maskA_step)[0]
        gain_A = np.where(~self._maskA_flat & maskA_step)[0]
        lost_B = np.where(self._maskB_flat & ~maskB_step)[0]
        gain_B = np.where(~self._maskB_flat & maskB_step)[0]

        freed_ids = np.concatenate([self._idA[lost_A], self._idB[lost_B]])
        freed_pos = np.vstack([refA[lost_A], refB[lost_B]]) if len(freed_ids) else np.zeros((0, 3))
        gained_pos = np.vstack([refA[gain_A], refB[gain_B]]) if (len(gain_A) + len(gain_B)) else np.zeros((0, 3))

        if len(freed_ids) != len(gained_pos):
            raise ValueError(
                "Atomic construction failed for image %d: %d sites were vacated but %d "
                "were filled, so the atom IDs of the flat GB and this image cannot be "
                "put in one-to-one correspondence.  The disconnection loop probably does "
                "not span a whole number of CSL periods."
                % (image_number, len(freed_ids), len(gained_pos)))

        order_freed  = np.lexsort((freed_pos[:, 0], freed_pos[:, 1], freed_pos[:, 2]))
        order_gained = np.lexsort((gained_pos[:, 0], gained_pos[:, 1], gained_pos[:, 2]))
        transferred = np.empty(len(gained_pos))
        transferred[order_gained] = freed_ids[order_freed]

        idA = self._idA.copy()
        idB = self._idB.copy()
        idA[gain_A] = transferred[:len(gain_A)]
        idB[gain_B] = transferred[len(gain_A):]

        gA = np.column_stack([refA[maskA_step], idA[maskA_step]])
        gB = np.column_stack([refB[maskB_step], idB[maskB_step]])

        # --- Apply the plastic (solid angle) displacement -----------------------
        for grain in (gA, gB):
            for i in range(len(grain)):
                point = np.array([grain[i, 1], grain[i, 0]])
                grain[i, 1] -= bicrystal._apply_plastic_displacement(
                    nodes_modified, period, burgers_vector, point, box[1, 1], -box[1, 1])

        self.grain1 = gA
        self.grain2 = gB

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
    def _unique_atoms(atoms, tol=None):
        """
            Removes lattice sites that were generated more than once.

            Args:
                atoms (np.ndarray): (N,3) array of positions.
                tol (float, optional): Separation below which two entries are the same
                    site. Defaults to `_EPS_DUPLICATE` (1e-3 A).

            Returns:
                np.ndarray: `atoms` with duplicates dropped, original order preserved.

            Notes:
                Uses a KD-tree rather than the O(N^2) pairwise loop this replaced.  That
                loop also compared a 0.5 A per-component *box* rather than a distance,
                which is large enough to delete genuinely distinct atoms in a lattice
                with a small interplanar spacing.
        """
        if tol is None:
            tol = bicrystal._EPS_DUPLICATE
        if len(atoms) == 0:
            return atoms
        mask = np.ones(len(atoms), dtype=bool)
        for i, j in cKDTree(atoms).query_pairs(tol):
            mask[max(i, j)] = False
        return atoms[mask]

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
