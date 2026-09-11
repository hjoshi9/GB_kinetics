import numpy as np
from scipy.spatial import cKDTree


class shuffle_pattern:
    """
        The elementary shuffle that moves the GB by one step, in a form that can be
        stamped onto a system of any size.

        The transformation a disconnection leaves behind it is the same everywhere
        inside its loop: one slab of grain A has become grain B. That transformation
        is periodic in the GB plane -- period `p` along the GB period direction and
        the CSL repeat `c` along the tilt axis -- so the whole of it is contained in a
        single CSL cell. Solving optimal transport on a larger system re-derives the
        same cell over and over, at a cost that grows as the square of the atom count.

        This class holds the displacement field of that one cell, keyed by where an
        atom sits inside it. `min_shuffle.map_from_reference` looks a displacement up
        for every atom of a large system instead of solving transport again.

        Attributes:
            keys (np.ndarray): (M,3) reduced positions, the lookup coordinates. Column
                0 is x measured from the GB, columns 1 and 2 are y and z folded into
                one CSL cell.
            displacements (np.ndarray): (M,3) shuffle displacement at each key.
            period (float): CSL period along the GB (`p`), in angstroms.
            tilt_repeat (float): CSL repeat along the tilt axis (`c`), in angstroms.
            weight (float): Weight this branch carried in the reference solve.
            branch (int): Index of the reference branch this pattern came from.
            spread (float): Largest disagreement, in angstroms, among the copies of
                a site that voted together. Zero for a pattern that is cleanly one
                CSL cell repeated; see `_condense`.
            consensus (float): Smallest share of a site's copies that backed the
                displacement chosen for it. 1.0 when every site was unanimous.
            split_sites (int): How many sites were not unanimous.
            reg_used (float): Regularization the reference solve converged at.
            gb_location (float): x of the GB in the system this was solved on. Keys
                measure x from it, so anything looking a displacement up in the
                reference's own frame -- a replica of it, say -- must key against this
                and not against the nominal GB plane.
    """

    # Two atoms closer than this once folded into the cell are the same site.
    _KEY_TOLERANCE = 0.5
    # Copies of a site whose displacements differ by less than this (A) voted the
    # same way. Well under the nearest-neighbour distance, well over relaxation.
    _AGREE_TOLERANCE = 0.5
    # Above this spread (A) even the copies that agreed do not really agree.
    _SPREAD_LIMIT = 0.5
    # A site carried by this share of its copies or less has no majority (a 2-2 tie
    # included), so which mechanism the cell holds would come down to the solver's
    # arbitrary split rather than to the structure.
    _CONSENSUS_LIMIT = 0.5

    def __init__(self, keys, displacements, period, tilt_repeat,
                 weight=1.0, branch=0, spread=0.0, reg_used=None,
                 consensus=1.0, split_sites=0, gb_location=0.0):
        self.keys = keys
        self.displacements = displacements
        self.period = period
        self.tilt_repeat = tilt_repeat
        self.weight = weight
        self.branch = branch
        self.spread = spread
        self.reg_used = reg_used
        self.consensus = consensus
        self.split_sites = split_sites
        self.gb_location = gb_location
        # check() is called once per image; the verdict is about the reference, so it
        # is worth saying once.
        self._reported = False
        self._tree = None
        self._tree_disp = None

    @staticmethod
    def reduce(positions, gb_location, period, tilt_repeat):
        """
            Folds positions into one CSL cell to get lookup keys.

            Args:
                positions (np.ndarray): (N,3) atomic positions.
                gb_location (float): x of the GB plane, which x is measured from so
                    that systems whose GB relaxed to slightly different places still
                    key the same.
                period (float): CSL period along the GB.
                tilt_repeat (float): CSL repeat along the tilt axis.

            Returns:
                np.ndarray: (N,3) keys.
        """
        keys = np.empty_like(positions, dtype=float)
        keys[:, 0] = positions[:, 0] - gb_location
        keys[:, 1] = np.mod(positions[:, 1], period)
        keys[:, 2] = np.mod(positions[:, 2], tilt_repeat)
        return keys

    @classmethod
    def from_solution(cls, Xs, Ys, gb_location, period, tilt_repeat, box,
                      weight=1.0, branch=0, reg_used=None):
        """
            Builds a pattern from one branch of a solved reference system.

            Args:
                Xs (np.ndarray): Initial atom rows of a `min_shuffle` branch; columns
                    0:3 are positions.
                Ys (np.ndarray): Where those atoms shuffle to, wrapped back into the
                    reference box.
                gb_location (float): x of the GB plane in the reference system.
                period (float): CSL period along the GB.
                tilt_repeat (float): CSL repeat along the tilt axis.
                box (np.ndarray): (3,2) reference shuffle box. `Ys` is wrapped into
                    it, so an atom that shuffled across a face reads back as having
                    crossed the whole box; the displacement is taken as the minimum
                    image along the two periodic directions to undo that.
                weight (float): Weight of this branch in the reference solve.
                branch (int): Index of this branch.
                reg_used (float): Regularization the reference solve converged at.

            Returns:
                shuffle_pattern: The condensed elementary cell.
        """
        X = np.asarray(Xs, dtype=float)[:, :3]
        Y = np.asarray(Ys, dtype=float)[:, :3]
        disp = Y - X
        lengths = np.asarray(box, dtype=float)[:, 1] - np.asarray(box, dtype=float)[:, 0]
        for axis in (1, 2):                      # y and z are the periodic ones
            disp[:, axis] -= lengths[axis] * np.round(disp[:, axis] / lengths[axis])
        keys = cls.reduce(X, gb_location, period, tilt_repeat)
        keys, disp, spread, consensus, split_sites = cls._condense(keys, disp, period,
                                                                   tilt_repeat)
        return cls(keys, disp, period, tilt_repeat, weight, branch, spread, reg_used,
                   consensus, split_sites, gb_location)

    @classmethod
    def _condense(cls, keys, displacements, period, tilt_repeat):
        """
            Merges the copies of each site into the one displacement they agree on.

            A reference system holds several copies of the CSL cell, translations of
            one another, so a site's copies must shuffle identically. Where they do
            not, the transport problem had more than one optimum and the solver split
            the copies between them -- a genuinely different mechanism is what the
            branches are for, not something a single cell should average over. So the
            displacement most copies voted for wins, and the dissent is recorded.

            Args:
                keys (np.ndarray): (N,3) reduced positions.
                displacements (np.ndarray): (N,3) displacement at each.
                period (float): CSL period along the GB.
                tilt_repeat (float): CSL repeat along the tilt axis.

            Returns:
                tuple: (keys, displacements, spread, consensus, split_sites) with one
                    row per distinct site, the largest disagreement inside a winning
                    vote, the smallest majority any site was carried by, and how many
                    sites were not unanimous.
        """
        groups = cls._group(keys, period, tilt_repeat)
        out_keys = np.zeros((len(groups), 3))
        out_disp = np.zeros((len(groups), 3))
        spread = 0.0
        consensus = 1.0
        split_sites = 0
        for i, members in enumerate(groups):
            d = displacements[members]
            winner, backing = cls._vote(d)
            out_keys[i] = keys[members[0]]
            out_disp[i] = winner
            if len(backing) > 1:
                spread = max(spread, float(np.abs(d[backing] - winner).max()))
            if len(backing) < len(members):
                split_sites += 1
                consensus = min(consensus, len(backing) / len(members))
        return out_keys, out_disp, spread, consensus, split_sites

    @classmethod
    def _vote(cls, d):
        """
            The displacement the most copies of a site agree on.

            Args:
                d (np.ndarray): (k,3) displacements of one site's copies.

            Returns:
                tuple: (winner, backing) -- the mean of the largest cluster of
                    mutually-agreeing copies, and their row indices. Ties go to the
                    smaller displacement, so the choice does not depend on ordering.
        """
        best_rows = None
        best_disp = None
        for i in range(len(d)):
            rows = np.where(np.abs(d - d[i]).max(axis=1) <= cls._AGREE_TOLERANCE)[0]
            candidate = d[rows].mean(axis=0)
            if (best_rows is None or len(rows) > len(best_rows)
                    or (len(rows) == len(best_rows)
                        and np.linalg.norm(candidate) < np.linalg.norm(best_disp))):
                best_rows, best_disp = rows, candidate
        return best_disp, best_rows

    @staticmethod
    def _group(keys, period, tilt_repeat):
        """
            Groups keys that name the same site, wrapping y and z.

            Args:
                keys (np.ndarray): (N,3) reduced positions.
                period (float): CSL period along the GB.
                tilt_repeat (float): CSL repeat along the tilt axis.

            Returns:
                list of np.ndarray: Row indices of `keys`, one array per site.
        """
        tree = cKDTree(keys, boxsize=None)
        # y and z are periodic in the folded cell, so a site sitting on the 0/period
        # face has neighbours at the far face. Query against shifted copies to catch
        # them rather than relying on cKDTree's boxsize, which needs x periodic too.
        shifts = np.array([[0, dy * period, dz * tilt_repeat]
                           for dy in (-1, 0, 1) for dz in (-1, 0, 1)])
        seen = np.zeros(len(keys), dtype=bool)
        groups = []
        for i in range(len(keys)):
            if seen[i]:
                continue
            members = set()
            for s in shifts:
                members.update(tree.query_ball_point(keys[i] + s,
                                                     shuffle_pattern._KEY_TOLERANCE))
            members = np.array(sorted(m for m in members if not seen[m]))
            seen[members] = True
            groups.append(members)
        return groups

    def lookup(self, positions, gb_location):
        """
            Displacement of the elementary shuffle at each of `positions`.

            Args:
                positions (np.ndarray): (N,3) atomic positions in the target system.
                gb_location (float): x of the GB plane in the target system.

            Returns:
                tuple: (displacements, distances) -- the (N,3) displacements and, per
                    atom, how far in angstroms it sat from the reference site it was
                    keyed to. Large values mean the target is not the reference system
                    repeated; they also mark atoms outside the slab the reference
                    shuffle was solved over, whose displacement is meaningless.
        """
        if self._tree is None:
            # Pad with the eight wrapped copies so a target atom near a cell face
            # matches the reference site across the face.
            shifts = np.array([[0, dy * self.period, dz * self.tilt_repeat]
                               for dy in (-1, 0, 1) for dz in (-1, 0, 1)])
            padded = np.vstack([self.keys + s for s in shifts])
            self._tree = cKDTree(padded)
            self._tree_disp = np.vstack([self.displacements] * len(shifts))
        keys = self.reduce(np.asarray(positions, dtype=float), gb_location,
                           self.period, self.tilt_repeat)
        dist, idx = self._tree.query(keys, k=1)
        return self._tree_disp[idx], dist

    def check(self, raise_on_failure=True):
        """
            Reports whether this pattern really is one CSL cell repeated.

            Args:
                raise_on_failure (bool): Raise instead of returning False.

            Returns:
                bool: True if the copies of the cell in the reference solve agreed
                    well enough for one of them to stand for all.

            Raises:
                ValueError: If they did not and `raise_on_failure` is set.
        """
        problem = None
        if self.spread > self._SPREAD_LIMIT:
            problem = ("copies that voted together still disagree by up to %.2f A"
                       % self.spread)
        elif self.consensus <= self._CONSENSUS_LIMIT:
            problem = ("%d site(s) had no majority -- the closest was carried by only "
                       "%.0f%% of its copies" % (self.split_sites, 100 * self.consensus))
        if problem is None:
            if self.split_sites and not self._reported:
                self._reported = True
                # Expected: a transport problem with several optima splits equivalent
                # copies between them. The majority is the cell; the alternatives are
                # what the other branches hold.
                print("   Reference branch %d: %d of %d site(s) were not unanimous, "
                      "carried by %.0f%% of their copies; majority taken."
                      % (self.branch, self.split_sites, len(self.keys),
                         100 * self.consensus))
            return True
        message = (
            "Reference branch %d is not a single CSL cell repeated: %s. Stamping it "
            "onto a larger system would impose one copy's shuffle everywhere. Lower "
            "regularizationParameter so the reference solve picks one mechanism, or "
            "turn off map_shuffle_from_reference." % (self.branch, problem))
        if raise_on_failure:
            raise ValueError(message)
        print("WARNING: " + message)
        return False

    def __repr__(self):
        return ("shuffle_pattern(branch=%d, sites=%d, weight=%.3f, spread=%.3f A, "
                "consensus=%.2f)" % (self.branch, len(self.keys), self.weight,
                                     self.spread, self.consensus))
