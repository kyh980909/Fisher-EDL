import importlib
import os
import sys
from contextlib import contextmanager


@contextmanager
def _temp_sys_path_without_repo(repo_root):
    original = list(sys.path)
    try:
        filtered = []
        repo_root = os.path.abspath(repo_root)
        for p in original:
            abs_p = os.path.abspath(p or os.getcwd())
            if abs_p == repo_root:
                continue
            filtered.append(p)
        sys.path[:] = filtered
        yield
    finally:
        sys.path[:] = original


def import_wandb(repo_root="."):
    # Avoid loading local ./wandb run directory as a namespace package.
    if "wandb" in sys.modules:
        mod = sys.modules["wandb"]
        if hasattr(mod, "init"):
            return mod
        del sys.modules["wandb"]

    with _temp_sys_path_without_repo(repo_root):
        wandb = importlib.import_module("wandb")

    if not hasattr(wandb, "init"):
        raise RuntimeError(
            "Failed to import official wandb SDK. "
            "Install wandb (`pip install wandb`) and ensure local `wandb/` dir is not shadowing imports."
        )
    return wandb
