import importlib
import os
import sys
from contextlib import contextmanager


@contextmanager
def _without_local_wandb_shadow():
    # Avoid local repository `wandb/` directory shadowing the pip package.
    cwd = os.getcwd()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    targets = {"", cwd, repo_root}

    removed = []
    for idx in reversed(range(len(sys.path))):
        path = sys.path[idx]
        norm = os.path.abspath(path) if path else ""
        if path in targets or norm in targets:
            removed.append((idx, path))
            sys.path.pop(idx)

    old_mod = sys.modules.pop("wandb", None)
    try:
        yield
    finally:
        for idx, path in sorted(removed, key=lambda x: x[0]):
            sys.path.insert(idx, path)
        if old_mod is not None:
            sys.modules["wandb"] = old_mod


def import_wandb():
    with _without_local_wandb_shadow():
        module = importlib.import_module("wandb")
    return module
