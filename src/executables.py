"""
    Locates the LAMMPS and MPI executables installed on the machine, so the driver
    scripts do not have to hardcode a path.

    More than a ``shutil.which`` call, because a binary that starts is not
    necessarily usable: a build without MANYBODY has no ``eam/alloy`` and one
    without REPLICA has no ``neb``, and such a build is often first on ``PATH``.
    Packaged MPI builds also ship broken library paths, and a serial build under
    ``mpirun -np N`` silently runs N copies of the same job rather than failing.

    So each candidate is probed with ``lmp -h`` for the styles it advertises, and
    the MPI one additionally with a two-rank ``-partition 2x1`` run. First to pass
    wins, letting a full build further down ``PATH`` beat a stripped one.

    Search order: the override variable (``GBK_LMP_SERIAL``, ``GBK_LMP_MPI``,
    ``GBK_MPIRUN``), then ``lammps_location`` / ``mpi_location``, then ``PATH``,
    then the standard install directories in ``_EXTRA_DIRS``.

    Run ``python -m src.executables`` to see what is selected here.
"""

import os
import shutil
import subprocess
import tempfile

# Names used by source builds, Homebrew, Debian/Ubuntu, Fedora and conda-forge.
# Generic names come last so that an explicitly flavoured binary is preferred.
_SERIAL_NAMES = ("lmp_serial", "lmp_g++_serial", "lmp_mac", "lmp_ubuntu", "lmp", "lammps")
_MPI_NAMES = ("lmp_mpi", "lmp_openmpi", "lmp_mpich", "lmp_g++_openmpi", "lmp_mac_mpi",
              "lmp", "lammps")
_MPIRUN_NAMES = ("mpirun", "mpiexec")

# Searched after PATH, for the common case of an MPI or LAMMPS install that is
# packaged outside the default PATH.
_EXTRA_DIRS = (
    "/opt/homebrew/bin",             # Homebrew, Apple silicon
    "/usr/local/bin",                # Homebrew, Intel macOS; source installs
    "/usr/bin",
    "/usr/lib64/openmpi/bin",        # Fedora / RHEL MPI packages
    "/usr/lib64/mpich/bin",
    "/usr/lib/x86_64-linux-gnu/openmpi/bin",   # Debian / Ubuntu
    "/usr/lib/x86_64-linux-gnu/mpich/bin",
)

# Full paths given here override discovery entirely.
_ENV_SERIAL = "GBK_LMP_SERIAL"
_ENV_MPI = "GBK_LMP_MPI"
_ENV_MPIRUN = "GBK_MPIRUN"

#: Styles needed to run the minimization inputs; stands in for MANYBODY.
STYLES_MINIMIZATION = ("eam/alloy",)

#: Styles needed to run the NEB inputs; ``neb`` stands in for REPLICA.
STYLES_NEB = ("eam/alloy", "neb")

_probe_cache = {}


def _from_env(variable):
    """
        Returns the executable named by an override environment variable.

        Args:
            variable (str): Name of the environment variable to read.

        Returns:
            str or None: The path, if the variable is set.

        Raises:
            RuntimeError: If set but not naming an executable file.
    """
    path = os.environ.get(variable)
    if not path:
        return None
    path = os.path.expanduser(path)
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        raise RuntimeError("%s is set to '%s', which is not an executable file."
                           % (variable, path))
    return path


def _candidates(names, hint=None, extra_dirs=_EXTRA_DIRS):
    """
        Builds the ordered list of executables to probe.

        Args:
            names (tuple of str): Executable names to look for, most specific first.
            hint (str, optional): Directory the user supplied. Searched before PATH.
            extra_dirs (tuple of str): Directories searched after PATH.

        Returns:
            list of str: Absolute paths that exist and are executable, in priority
                order, without duplicates.
    """
    dirs = ([hint] if hint else []) + list(extra_dirs)
    found = []
    for name in names:
        for directory in dirs:
            path = os.path.join(os.path.expanduser(directory), name)
            if os.path.isfile(path) and os.access(path, os.X_OK):
                found.append(os.path.realpath(path))
        on_path = shutil.which(name)
        if on_path:
            found.append(os.path.realpath(on_path))
    # A hinted directory outranks PATH; among equals, order is by name specificity.
    return list(dict.fromkeys(found))


def _lammps_styles(path):
    """
        Runs `path -h` and returns the styles the build advertises.

        Args:
            path (str): Absolute path to a candidate LAMMPS executable.

        Returns:
            set of str or None: Tokens of the help output, which include every
                installed style and command name, or None if the executable could
                not be run at all.
    """
    if path in _probe_cache:
        return _probe_cache[path]
    try:
        # `-log none`: LAMMPS opens log.lammps on startup even for `-h`.
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([path, "-h", "-log", "none"], cwd=tmp,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=60)
        styles = set(result.stdout.decode(errors="replace").split()) if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        styles = None
    _probe_cache[path] = styles
    return styles


def _runs_in_parallel(mpirun, lammps):
    """
        Checks that `mpirun` and `lammps` together give a real multi-partition run.

        Uses the same `-partition Nx1` call as `run_neb_calc`, on an empty input, so
        serial builds and broken library paths are rejected before any input files
        are written.

        Args:
            mpirun (str): Absolute path to mpirun/mpiexec.
            lammps (str): Absolute path to a candidate LAMMPS executable.

        Returns:
            bool: True if a two-rank, two-partition run succeeded.
    """
    key = ("parallel", mpirun, lammps)
    if key in _probe_cache:
        return _probe_cache[key]
    ok = False
    with tempfile.TemporaryDirectory() as tmp:
        empty = os.path.join(tmp, "in.probe")
        open(empty, "w").close()
        base = [lammps, "-partition", "2x1", "-in", empty, "-log", "none", "-screen", "none"]
        # --oversubscribe is Open MPI only; MPICH rejects it, so fall back.
        for launcher in ([mpirun, "--oversubscribe", "-np", "2"], [mpirun, "-np", "2"]):
            try:
                result = subprocess.run(launcher + base, cwd=tmp, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, timeout=120)
            except (OSError, subprocess.SubprocessError):
                continue
            if result.returncode == 0:
                ok = True
                break
    _probe_cache[key] = ok
    return ok


def _describe(names, hint, required_styles):
    """
        Builds the error message shown when nothing usable was found.

        Args:
            names (tuple of str): Executable names that were searched for.
            hint (str or None): Directory the user supplied, if any.
            required_styles (tuple of str): Styles the build had to provide.

        Returns:
            str: A message naming what was looked for and what was rejected.
    """
    lines = ["  looked for: " + ", ".join(names)]
    if hint:
        lines.append("  in the directory you gave: " + hint)
    lines.append("  then on PATH, then in: " + ", ".join(_EXTRA_DIRS))
    wrong_styles, would_not_start = [], []
    for path in (_candidates(names, hint) if required_styles else []):
        styles = _lammps_styles(path)
        if styles is None:
            would_not_start.append(path)
        elif not all(s in styles for s in required_styles):
            missing = [s for s in required_styles if s not in styles]
            wrong_styles.append("%s (no %s)" % (path, ", ".join(missing)))
    if wrong_styles:
        lines.append("  binaries missing a required style: " + ", ".join(wrong_styles))
    if would_not_start:
        lines.append("  binaries that would not start at all: " + ", ".join(would_not_start))
    return "\n".join(lines)


def find_lammps_serial(hint=None, required_styles=STYLES_MINIMIZATION):
    """
        Finds a LAMMPS executable able to run the minimization inputs.

        Args:
            hint (str, optional): Directory to search before PATH; None searches
                only PATH and the standard install directories.
            required_styles (tuple of str): Styles the build must advertise.

        Returns:
            str: Absolute path to the executable.

        Raises:
            RuntimeError: If no candidate both runs and provides `required_styles`.
    """
    override = _from_env(_ENV_SERIAL)
    if override:
        return override
    for path in _candidates(_SERIAL_NAMES, hint):
        styles = _lammps_styles(path)
        if styles is not None and all(s in styles for s in required_styles):
            return path
    raise RuntimeError(
        "No usable LAMMPS executable found.  It must provide: %s\n%s\n"
        "  Set %s to the full path of the binary to use, or pass lammps_location."
        % (", ".join(required_styles), _describe(_SERIAL_NAMES, hint, required_styles), _ENV_SERIAL))


def find_mpirun(hint=None):
    """
        Finds an MPI launcher.

        Args:
            hint (str, optional): Directory to search before PATH.

        Returns:
            str: Absolute path to mpirun or mpiexec.

        Raises:
            RuntimeError: If neither is found.
    """
    override = _from_env(_ENV_MPIRUN)
    if override:
        return override
    for path in _candidates(_MPIRUN_NAMES, hint):
        return path
    raise RuntimeError(
        "No MPI launcher found.\n%s\n  Set %s to the full path of mpirun, or pass mpi_location."
        % (_describe(_MPIRUN_NAMES, hint, ()), _ENV_MPIRUN))


def find_lammps_mpi(mpirun, hint=None, required_styles=STYLES_NEB):
    """
        Finds a LAMMPS executable that is MPI-enabled and has the needed styles.

        Args:
            mpirun (str): Launcher to test against, since an MPI build only works
                under the implementation it was compiled against.
            hint (str, optional): Directory to search before PATH.
            required_styles (tuple of str): Styles the build must advertise.

        Returns:
            str: Absolute path to the executable.

        Raises:
            RuntimeError: If no candidate passes both the style and the parallel check.
    """
    override = _from_env(_ENV_MPI)
    if override:
        return override
    lacked_parallel = []
    for path in _candidates(_MPI_NAMES, hint):
        styles = _lammps_styles(path)
        if styles is None or not all(s in styles for s in required_styles):
            continue
        if _runs_in_parallel(mpirun, path):
            return path
        lacked_parallel.append(path)
    message = ["No usable MPI-enabled LAMMPS executable found.  It must provide: %s"
               % ", ".join(required_styles),
               _describe(_MPI_NAMES, hint, required_styles)]
    if lacked_parallel:
        message.append("  binaries with the right styles that are not MPI builds (or whose "
                       "MPI does not match %s): %s" % (mpirun, ", ".join(lacked_parallel)))
    message.append("  Set %s to the full path of the binary to use, or pass lammps_location."
                   % _ENV_MPI)
    raise RuntimeError("\n".join(message))


def report(hint_lammps=None, hint_mpi=None):
    """
        Prints what discovery finds, for checking a machine before a long run.

        Args:
            hint_lammps (str, optional): Directory hint for the LAMMPS executables.
            hint_mpi (str, optional): Directory hint for the MPI launcher.

        Returns:
            dict: Keys `serial`, `mpirun` and `mpi`, each an absolute path or the
                error message explaining why nothing was found.
    """
    found = {}
    for key, finder in (("serial", lambda: find_lammps_serial(hint_lammps)),
                        ("mpirun", lambda: find_mpirun(hint_mpi))):
        try:
            found[key] = finder()
        except RuntimeError as error:
            found[key] = str(error)
    try:
        found["mpi"] = (find_lammps_mpi(found["mpirun"], hint_lammps)
                        if os.path.isfile(str(found["mpirun"])) else "no MPI launcher")
    except RuntimeError as error:
        found["mpi"] = str(error)
    for key in ("serial", "mpi", "mpirun"):
        print("%-8s : %s" % (key, found[key]))
    return found


if __name__ == "__main__":
    report()
