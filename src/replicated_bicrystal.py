import numpy as np

from src.plastic_slip import dislocation_dipole


class replicated_bicrystal:
    """
        Builds a large system's images by replicating a relaxed reference cell.

        The other route to a disconnection image builds it from the reference lattice
        and hands the result to LAMMPS: every atom starts on an ideal site with an
        analytic slip field laid over it, so the minimiser has to find the whole
        relaxation from scratch, at every image, for a system that is mostly a repeat
        of a much smaller one.

        Here the smallest bicrystal is relaxed once and its atoms are tiled outwards
        instead. The boundary of the tiled system is then stepped by laying down two
        fields: inside the disconnection loop, atoms near the GB take the elementary
        shuffle held in a `shuffle_pattern`, and everything else takes the plastic
        displacement of the dislocation dipole. What comes out is a structure that is
        already relaxed everywhere except at the two cores.

        Attributes:
            atoms (np.ndarray): (N,5) replicated flat GB -- id, type, x, y, z.
            box (np.ndarray): (3,2) box of the replicated system.
            reference_box (np.ndarray): (3,2) box of the cell that was replicated.
            period (float): CSL period along the GB.
            n_period (int): Replication factor along the GB period.
            n_tilt (int): Replication factor along the tilt axis.
    """

    # An atom further than this (A) from any site of the reference cell is not in the
    # region the shuffle pattern covers, so it takes the plastic field instead.
    _DOMAIN_CUTOFF = 1.0

    def __init__(self, reference_atoms, reference_box, period, n_period, n_tilt):
        """
            Tiles a relaxed reference cell out to the requested size.

            Args:
                reference_atoms (np.ndarray): (M,5) atoms of the relaxed reference
                    system -- id, type, x, y, z.
                reference_box (np.ndarray): (3,2) box of that system.
                period (float): CSL period along the GB, in angstroms.
                n_period (int): How many reference cells along the GB period.
                n_tilt (int): How many along the tilt axis.
        """
        self.reference_box = np.asarray(reference_box, dtype=float)
        self.period = period
        self.n_period = n_period
        self.n_tilt = n_tilt
        atoms, box = self._tile(np.asarray(reference_atoms, dtype=float),
                                self.reference_box, n_period, n_tilt)
        self.atoms = atoms
        self.box = box

    @staticmethod
    def _tile(atoms, box, n_period, n_tilt):
        """
            Repeats a cell outwards from its centre along y and z.

            Copies are grown outwards rather than stacked from one face: the half of
            the cell below the centre is pushed down and the half above is pushed up,
            so the original cell stays where it was and the disconnection stays in the
            middle of the box. Stacking from a face would slide the boundary off
            centre as the system grew.

            Args:
                atoms (np.ndarray): (M,5) atoms -- id, type, x, y, z.
                box (np.ndarray): (3,2) box of the cell.
                n_period (int): Replication factor along y.
                n_tilt (int): Replication factor along z.

            Returns:
                tuple: (atoms, box) of the replicated system, atoms renumbered 1..N in
                    the order they were generated.
        """
        out = atoms
        for axis, n in ((3, n_period), (4, n_tilt)):
            if n <= 1:
                continue
            column = axis - 2                      # atom column 3 is y, box row 1
            length = box[column, 1] - box[column, 0]
            middle = 0.5 * (box[column, 0] + box[column, 1])
            copies = [out]
            for i in range(1, n):
                shifted = out.copy()
                below = shifted[:, axis] < middle
                shifted[below, axis] -= i * length / 2.0
                shifted[~below, axis] += i * length / 2.0
                copies.append(shifted)
            out = np.vstack(copies)
            box = box.copy()
            box[column, :] *= n
        out = out.copy()
        out[:, 0] = np.arange(1, len(out) + 1)
        return out, box

    @staticmethod
    def plastic_displacement(nodes, period, burgers_vector, points, boxlimhi, boxlimlo,
                             nImages=2):
        """
            Plastic displacement of a set of dipoles, evaluated at many points at once.

            Same quantity as `bicrystal._apply_plastic_displacement`, and the same
            solid-angle formula, but over an array of points rather than one at a
            time. Building an image means evaluating this for every atom, so the
            per-point version dominates the construction.

            Args:
                nodes (list): Node pairs defining the dipoles, as
                    `create_disconnection_containing_bicrystal` builds them.
                period (float): Periodicity length along the boundary.
                burgers_vector (float): Burgers vector of the dislocation.
                points (np.ndarray): (N,2) coordinates as (y, x), matching the
                    convention the scalar version takes.
                boxlimhi (float): Upper box limit along the GB period.
                boxlimlo (float): Lower box limit along the GB period.
                nImages (int): Periodic image dipoles either side.

            Returns:
                np.ndarray: (N,) displacement at each point.
        """
        points = np.asarray(points, dtype=float)
        angle = np.zeros(len(points))
        for dipole_nodes in nodes:
            dipole_nodes = np.asarray(dipole_nodes, dtype=float)
            separation = dipole_nodes[0] - dipole_nodes[1]
            length = np.linalg.norm(separation)
            if abs(length) < 1e-6:
                continue
            t = separation / length
            n = np.array([-t[1], t[0]])
            half_length = 0.5 * length
            centre = 0.5 * (dipole_nodes[0] + dipole_nodes[1])
            for image in range(-nImages, nImages + 1):
                shifted = points + image * period * 100
                local_t = (shifted - centre) @ t
                local_n = (shifted - centre) @ n
                # On the cut itself the arctangents are degenerate. The scalar version
                # adds nothing there unless the point is strictly off the plane, which
                # it cannot be, so the contribution is zero.
                off_plane = np.abs(local_n) >= 1e-16
                y_term = np.zeros(len(points))
                np.divide(1.0, np.abs(local_n), out=y_term, where=off_plane)
                contribution = 2 * np.sign(local_n) * (
                    -np.arctan((local_t - half_length) * y_term)
                    + np.arctan((local_t + half_length) * y_term))
                angle += np.where(off_plane, contribution, 0.0)
        displacement = -(angle / (4.0 * np.pi)) * burgers_vector
        box_length = boxlimhi - boxlimlo
        moved = points[:, 0] - displacement
        displacement = np.where(moved > boxlimhi, displacement + box_length, displacement)
        displacement = np.where(moved < boxlimlo, displacement - box_length, displacement)
        return displacement

    def build_image(self, nodes_modified, loop_start, loop_stop, pattern,
                    gb_position, step_height, burgers_vector, period):
        """
            Steps the boundary of the replicated system over one disconnection loop.

            Two fields are laid over the relaxed flat GB. Atoms close enough to the
            boundary for the reference cell to describe them, and inside the loop,
            take the elementary shuffle. Every other atom takes the dipole's plastic
            displacement, which is the rigid slip deep inside the loop and dies away
            outside it.

            Args:
                nodes_modified (list): Dipoles of this image, as
                    `create_disconnection_containing_bicrystal` assembles them.
                loop_start (float): y of the leading disconnection node.
                loop_stop (float): y of the trailing node.
                pattern (shuffle_pattern): The elementary shuffle.
                gb_position (float): x of the flat GB plane.
                step_height (float): Step height of the disconnection mode.
                burgers_vector (float): Burgers vector of the mode.
                period (float): CSL period along the GB.

            Returns:
                tuple: (atoms, shuffled, matched) -- the (N,5) image, how many atoms
                    took the shuffle, and how far the worst of them sat from the
                    reference site it was keyed to.
        """
        atoms = self.atoms.copy()
        positions = atoms[:, 2:5]
        # These atoms are the reference's own, tiled, so they are keyed in the
        # reference's frame -- against the GB where it actually relaxed to, not the
        # nominal plane.
        displacement, distance = pattern.lookup(positions, pattern.gb_location)

        inside = (positions[:, 1] >= loop_start) & (positions[:, 1] < loop_stop)
        # The pattern only covers the slab the reference shuffle was solved over.
        # Beyond it a lookup returns whatever site happened to be nearest, so those
        # atoms are handed to the plastic field instead.
        in_domain = distance <= self._DOMAIN_CUTOFF
        shuffles = inside & in_domain

        atoms[shuffles, 2:5] += displacement[shuffles]

        slip_points = np.column_stack([positions[~shuffles, 1], positions[~shuffles, 0]])
        slip = self.plastic_displacement(nodes_modified, period, burgers_vector,
                                         slip_points, self.box[1, 1], self.box[1, 0])
        atoms[~shuffles, 3] -= slip

        self._wrap(atoms)
        atoms[:, 1] = self._grain_types(atoms, gb_position, step_height,
                                        loop_start, loop_stop)
        matched = float(distance[shuffles].max()) if shuffles.any() else 0.0
        return atoms, int(shuffles.sum()), matched

    def build_flat(self, gb_position):
        """
            The replicated flat GB, with grain labels reapplied.

            Args:
                gb_position (float): x of the GB plane.

            Returns:
                np.ndarray: (N,5) atoms of the flat boundary.
        """
        atoms = self.atoms.copy()
        atoms[:, 1] = self._grain_types(atoms, gb_position, 0.0, 0.0, 0.0)
        return atoms

    @staticmethod
    def _grain_types(atoms, gb_position, step_height, loop_start, loop_stop):
        """
            Labels atoms by the grain they belong to, split on the stepped cut plane.

            Mirrors the cut `create_disconnection_containing_bicrystal` masks on:
            `gb_position + step_height` inside the loop and `gb_position` outside, with
            grain 1 above it.

            Args:
                atoms (np.ndarray): (N,5) atoms -- id, type, x, y, z.
                gb_position (float): x of the flat GB plane.
                step_height (float): Step height, either sign.
                loop_start (float): y of the leading node.
                loop_stop (float): y of the trailing node.

            Returns:
                np.ndarray: (N,) grain labels, 1 or 2.
        """
        in_loop = (atoms[:, 3] >= loop_start) & (atoms[:, 3] < loop_stop)
        cut = gb_position + np.where(in_loop, step_height, 0.0)
        return np.where(atoms[:, 2] >= cut - 1e-4, 1.0, 2.0)

    def _wrap(self, atoms):
        """
            Brings atoms back inside the periodic directions, in place.

            Args:
                atoms (np.ndarray): (N,5) atoms -- id, type, x, y, z.
        """
        for axis, row in ((3, 1), (4, 2)):
            lo, hi = self.box[row, 0], self.box[row, 1]
            length = hi - lo
            atoms[:, axis] = lo + np.mod(atoms[:, axis] - lo, length)

    def write(self, atoms, folder, elem, sigma, inclination, size, image_num):
        """
            Writes an image under the name the rest of the pipeline expects.

            Args:
                atoms (np.ndarray): (N,5) atoms -- id, type, x, y, z.
                folder (str): Output directory, ending in a separator.
                elem (str): Element symbol.
                sigma (int): Sigma value.
                inclination (float): GB inclination.
                size (int): Size along the GB period, for the filename.
                image_num (int): Image index.

            Returns:
                str: The filename written, without the folder.
        """
        file = ("data." + elem + "s" + str(sigma) + "inc" + str(inclination)
                + "_size" + str(size) + "disc" + str(image_num))
        eps = 0.1
        order = np.argsort(atoms[:, 0])
        with open(folder + file, "w") as f:
            f.write("# LAMMPS data file Sigma = %d, inclination = %f\n" % (sigma, inclination))
            f.write("#LAMMPS data file\n")
            f.write("%d atoms\n" % len(atoms))
            f.write("2 atom types\n")
            f.write("%0.10f %0.10f xlo xhi\n" % (self.box[0, 0] - eps, self.box[0, 1] + eps))
            f.write("%0.10f %0.10f ylo yhi\n" % (self.box[1, 0], self.box[1, 1]))
            f.write("%0.10f %0.10f zlo zhi\n" % (self.box[2, 0], self.box[2, 1]))
            f.write("0.0 0.0 0.0 xy xz yz\n\n")
            f.write("Atoms # atomic\n\n")
            for i in order:
                f.write("%d %d %0.10f %0.10f %0.10f\n"
                        % (atoms[i, 0], atoms[i, 1], atoms[i, 2], atoms[i, 3], atoms[i, 4]))
        print("Done writing replicated bicrystal " + folder + file)
        return file
