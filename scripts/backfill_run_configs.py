import argparse
import glob
import os
import pickle
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Backfill config.txt for existing runs")
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class _PickleModule:
    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("fisher_edl"):
                return type("Dummy", (), {})
            return super().find_class(module, name)

    dump = pickle.dump
    dumps = pickle.dumps
    load = pickle.load
    loads = pickle.loads


def _safe_load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        pass
    except ModuleNotFoundError:
        pass
    return torch.load(path, map_location="cpu", pickle_module=_PickleModule)


def main():
    args = parse_args()
    runs_dir = os.path.abspath(args.runs_dir)
    run_dirs = sorted(glob.glob(os.path.join(runs_dir, "cifar_*")))
    print(f"found {len(run_dirs)} run dirs")

    for run_dir in run_dirs:
        cfg_path = os.path.join(run_dir, "config.txt")
        if os.path.exists(cfg_path) and not args.overwrite:
            continue
        ckpt_path = os.path.join(run_dir, "best_val_acc.pt")
        if not os.path.exists(ckpt_path):
            continue
        ckpt = _safe_load_checkpoint(ckpt_path)
        cfg = ckpt.get("config")
        if cfg is None:
            continue
        with open(cfg_path, "w") as f:
            for key, value in cfg.__dict__.items():
                f.write(f"{key}: {value}\n")
        print(f"wrote {cfg_path}")


if __name__ == "__main__":
    main()
