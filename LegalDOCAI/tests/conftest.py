import os
import sys


def _ensure_project_root_on_path() -> None:
    """
    Ensure tests can import `backend.*` when pytest is invoked
    from repository root or other working directories.
    """
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(tests_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


_ensure_project_root_on_path()
