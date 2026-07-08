import os
import subprocess
import sys


def ensure_casa_data_dir():
    """
    Create CASA's default data directory before casatools is imported.

    CASA uses this directory for measures/config data. Creating it before
    importing casatools allows CASA/casaconfig to populate it when needed.
    """
    casa_data_dir = os.path.expanduser("~/.casa/data")
    os.makedirs(casa_data_dir, exist_ok=True)
    return casa_data_dir


def update_casa_data():
    """
    Explicitly populate/update CASA runtime data via casaconfig.

    This can download a large CASA data bundle, so callers should only run it
    when the user has requested a repair.
    """
    ensure_casa_data_dir()
    try:
        import casaconfig  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "CASA data repair requires casaconfig. Install the CASA extra with "
            "`pip install -e '.[casa]'`, then retry."
        ) from exc

    cmd = [sys.executable, "-m", "casaconfig", "--update-all"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "CASA data repair failed while running `python -m casaconfig --update-all`. "
            "Check network access and CASA/casaconfig installation."
        ) from exc


def import_casa_tools(fix_data=False):
    """
    Prepare CASA data, optionally repair it, then import casatools.
    """
    ensure_casa_data_dir()
    if fix_data:
        update_casa_data()

    try:
        from casatools import measures, simulator, table
    except ImportError as exc:
        raise ImportError(
            "make-ms requires casatools. Install it with `pip install -e '.[casa]'` "
            "or run this step in a CASA-capable environment."
        ) from exc

    return simulator, measures, table


def check_casa_observatory(measures_tool, observatory_name):
    """
    Verify that CASA can resolve the observatory data needed by make-ms.
    """
    try:
        return measures_tool.observatory(observatory_name)
    except Exception as exc:
        raise RuntimeError(
            f"CASA imported, but could not load observatory data for {observatory_name!r}. "
            "Try rerunning make-ms with `--fix_casa_data true` to populate/update "
            "CASA runtime data, then rerun this command."
        ) from exc
