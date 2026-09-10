"""
    Locates the LAMMPS and MPI executables that are actually installed on the
    machine, so that the driver scripts do not have to hardcode a path.

    Discovery is deliberately more than a ``shutil.which`` call.  A LAMMPS binary
    that exists and starts is not necessarily one this pipeline can use:

    * LAMMPS is built with a selectable set of packages.  A build without MANYBODY
      has no ``eam/alloy`` pair style, and one without REPLICA has no ``neb``
      command.  Such a stripped build is often the first ``lmp`` on ``PATH``.
    * Distribution MPI builds are frequently installed with a broken library path
      and fail before reaching ``main``.
    * A serial build launched under ``mpirun -np N`` does not fail.  It silently
      runs N independent copies of the same calculation.

    Every candidate is therefore probed with ``lmp -h``, which lists the styles the
    build provides, and the MPI executable is additionally probed with a two-rank
    ``-partition 2x1`` run, exactly what :meth:`src.runLAMMPS.run_LAMMPS.run_neb_calc`
    performs.  The first candidate that passes wins, so a fully featured build
    further down ``PATH`` beats a stripped one at the front of it.

    Search order for each executable:

    #. The override environment variable, if set: ``GBK_LMP_SERIAL``,
       ``GBK_LMP_MPI`` or ``GBK_MPIRUN``, each naming a binary directly.
    #. The directory passed as ``lammps_location`` / ``mpi_location``, if given.
    #. ``PATH``.
    #. The standard install directories listed in ``_EXTRA_DIRS`` below.

    Run ``python -m src.executables`` to print what discovery selects on the
    current machine.
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

#: Styles a LAMMPS build must advertise to run the minimization inputs.
#: ``eam/alloy`` stands in for the MANYBODY package.
STYLES_MINIMIZATION = ("eam/alloy",)

#: Styles a LAMMPS build must advertise to run the NEB inputs. ``neb`` stands in
#: for the REPLICA package, which MPI builds are often shipped without.
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
            RuntimeError: If the variable is set but does not name an executable file,
                which is worth catching here rather than as a shell error mid-run.
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
    # A hinted directory outranks PATH, but among equals the order above is by
    # name specificity, which is what we want.
    return list(dict.fromkeys(found))


def _lammps_styles(path):
    """
        Runs `path -h` and returns the styles the build advertises.

        Args:
            path (str): Absolute path to a candidate LAMMPS executable.

        Returns:
            set of str or None: The whitespace-separated tokens of the help output,
                which include every installed style and command name, or None if the
                executable could not be run at all (missing library, wrong
                architecture, not LAMMPS).
    """
    if path in _probe_cache:
        return _probe_cache[path]
    try:
        # `-log none` matters: LAMMPS opens log.lammps in the working directory on
        # startup, even for `-h`, and probing must not write into the user's cwd.
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

        Runs the same `-partition Nx1` invocation `run_neb_calc` uses, on an empty
        input.  A serial build reports "Processor partitions do not match number of
        allocated processors" and a build with a broken library path fails to start,
        so both are rejected here rather than after the input files are written.

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
            hint (str, optional): Directory to search before PATH. This is the
                `lammps_location` the driver scripts pass; None means search only
                PATH and the standard install directories.
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
            hint (str, optional): Directory to search before PATH, i.e. the
                `mpi_location` the driver scripts pass.

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
            mpirun (str): Absolute path to the launcher the executable will run under;
                the candidate is tested with this specific launcher, because an MPI
                build only works under the MPI implementation it was compiled against.
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
