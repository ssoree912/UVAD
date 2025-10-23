#!/usr/bin/env python3
"""
Utility script to launch multiple AppAE trainings with configurable seeds, pruning,
and GPU assignment. Designed to generate checkpoints compatible with model soup
experiments.
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = sys.executable


@dataclass
class RunSpec:
    seed: int
    run_name: str
    magnitude: float
    random: float
    random_seed: Optional[int]
    unprune_epoch: Optional[int]
    label: str
    gpu_id: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch multiple AppAE training runs with pruning options."
    )
    parser.add_argument("--dataset_name", required=True,
                        choices=["shanghaitech", "avenue", "ped2"])
    parser.add_argument("--uvadmode", required=True, choices=["merge", "partial"])

    parser.add_argument("--base_seeds", nargs="+", type=int, default=[],
                        help="Seeds for plain training runs (for model soup).")

    parser.add_argument("--magnitude_seed", type=int, default=None,
                        help="Seed for magnitude-pruned baseline run.")
    parser.add_argument("--magnitude_prune", type=float, default=0.05,
                        help="Pruning ratio for magnitude pruning (default: 0.05).")

    parser.add_argument("--random_prune", type=float, default=0.05,
                        help="Pruning ratio for random pruning (default: 0.05).")
    parser.add_argument("--random_seeds", nargs="+", type=int, default=[],
                        help="Random seeds for random pruning masks.")
    parser.add_argument("--random_unprune_epochs", nargs="+", type=int, default=[-1],
                        help="Epochs after which to unprune random runs (use -1 to keep pruned).")
    parser.add_argument("--skip_magnitude_run", action="store_true",
                        help="Skip launching the pure magnitude-pruned baseline run.")

    parser.add_argument("--gpus", nargs="+", type=int, default=None,
                        help="GPU IDs to cycle through. If omitted, runs execute on CPU.")
    parser.add_argument("--python", default=DEFAULT_PYTHON,
                        help="Python interpreter to use (default: current).")

    parser.add_argument("--save_root", default="artifacts", help="Checkpoint base directory.")
    parser.add_argument("--log_dir", default="logger", help="Directory for log files.")
    parser.add_argument("--config", default=None, help="Path to training config YAML.")
    parser.add_argument("--no_stream_logs", action="store_true",
                        help="Disable stdout logging from training processes.")
    parser.add_argument("--include_mot", action="store_true",
                        help="After each training, run fGMM (mode=mot) on the same GPU.")

    parser.add_argument("--extra_args", default="", help="Additional arguments for app runs.")
    parser.add_argument("--dry_run", action="store_true", help="Show commands without executing.")
    parser.add_argument("--workspace", default=str(PROJECT_ROOT),
                        help="Working directory for subprocesses.")
    return parser.parse_args()


def format_amount(amount: float) -> str:
    return f"{int(amount * 100):02d}"


def build_run_specs(args: argparse.Namespace) -> List[RunSpec]:
    specs: List[RunSpec] = []
    for seed in args.base_seeds:
        run_name = f"seed{seed}"
        specs.append(RunSpec(
            seed=seed,
            run_name=run_name,
            magnitude=0.0,
            random=0.0,
            random_seed=None,
            unprune_epoch=None,
            label="baseline"
        ))

    if args.magnitude_seed is not None:
        amt = format_amount(args.magnitude_prune)
        if not args.skip_magnitude_run:
            run_name = f"mag{amt}_seed{args.magnitude_seed}"
            specs.append(RunSpec(
                seed=args.magnitude_seed,
                run_name=run_name,
                magnitude=args.magnitude_prune,
                random=0.0,
                random_seed=None,
                unprune_epoch=None,
                label="magnitude"
            ))

        rand_amt = format_amount(args.random_prune)
        for rand_seed in args.random_seeds:
            for epoch in args.random_unprune_epochs:
                unprune_epoch = None if epoch < 0 else epoch
                suffix = "" if unprune_epoch is None else f"_unprune{epoch:02d}"
                run_name = (
                    f"mag{amt}_rand{rand_amt}_base{args.magnitude_seed}_r{rand_seed}{suffix}"
                )
                label = "rand_prune" if unprune_epoch is None else "rand_prune_unprune"
                specs.append(RunSpec(
                    seed=args.magnitude_seed,
                    run_name=run_name,
                    magnitude=args.magnitude_prune,
                    random=args.random_prune,
                    random_seed=rand_seed,
                    unprune_epoch=unprune_epoch,
                    label=label
                ))
    return specs


def acquire_free_gpu(running: List[dict], gpu_pool: List[Optional[int]]) -> Optional[int]:
    occupied = {entry["gpu"] for entry in running}
    for gpu_id in gpu_pool:
        if gpu_id not in occupied:
            return gpu_id
    return None


def launch_training(args: argparse.Namespace, spec: RunSpec, gpu_id: Optional[int]) -> subprocess.Popen:
    cmd: List[str] = [
        args.python,
        "main1_pseudoanomaly.py",
        "--dataset_name", args.dataset_name,
        "--uvadmode", args.uvadmode,
        "--mode", "app",
        "--seed", str(spec.seed),
        "--run_name", spec.run_name,
        "--magnitude_prune", str(spec.magnitude),
        "--random_prune", str(spec.random)
    ]
    if args.config:
        cmd += ["--config", args.config]
    if args.save_root:
        cmd += ["--save_root", args.save_root]
    if args.log_dir:
        cmd += ["--log_dir", args.log_dir]
    if spec.random_seed is not None:
        cmd += ["--random_prune_seed", str(spec.random_seed)]
    if spec.unprune_epoch is not None:
        cmd += ["--unprune_epoch", str(spec.unprune_epoch)]
    if args.no_stream_logs:
        cmd.append("--no_stream_logs")
    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    workdir = args.workspace
    print(f"[RUN] GPU={gpu_id if gpu_id is not None else 'cpu'} :: {' '.join(cmd)}")
    if args.dry_run:
        return subprocess.Popen(["true"])

    return subprocess.Popen(cmd, env=env, cwd=workdir)


def run_mot_stage(args: argparse.Namespace, spec: RunSpec, gpu_id: Optional[int]) -> int:
    cmd: List[str] = [
        args.python,
        "main1_pseudoanomaly.py",
        "--dataset_name", args.dataset_name,
        "--uvadmode", args.uvadmode,
        "--mode", "mot",
        "--seed", str(spec.seed),
        "--run_name", f"{spec.run_name}_mot",
    ]
    if args.config:
        cmd += ["--config", args.config]
    if args.save_root:
        cmd += ["--save_root", args.save_root]
    if args.log_dir:
        cmd += ["--log_dir", args.log_dir]
    if args.no_stream_logs:
        cmd.append("--no_stream_logs")
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    workdir = args.workspace
    print(f"[RUN] GPU={gpu_id if gpu_id is not None else 'cpu'} :: {' '.join(cmd)}")
    if args.dry_run:
        return 0
    result = subprocess.run(cmd, env=env, cwd=workdir)
    return result.returncode


def main():
    args = parse_args()
    specs = build_run_specs(args)
    if not specs:
        print("No runs scheduled. Provide --base_seeds or pruning options.", file=sys.stderr)
        sys.exit(1)

    gpu_pool = args.gpus if args.gpus else [None]
    queue: Deque[RunSpec] = deque(specs)
    running: List[dict] = []

    while queue or running:
        while queue:
            gpu_id = acquire_free_gpu(running, gpu_pool)
            if gpu_id is None:
                break
            spec = queue.popleft()
            proc = launch_training(args, spec, gpu_id)
            running.append({"proc": proc, "spec": spec, "gpu": gpu_id})

        time.sleep(1)
        for entry in list(running):
            proc: subprocess.Popen = entry["proc"]
            if proc.poll() is None:
                continue
            running.remove(entry)
            rc = proc.returncode
            spec = entry["spec"]
            gpu_id = entry["gpu"]
            status = "OK" if rc == 0 else f"FAIL({rc})"
            print(f"[DONE] {spec.run_name} status={status}")
            if rc != 0:
                continue
            if args.include_mot:
                mot_rc = run_mot_stage(args, spec, gpu_id)
                mot_status = "OK" if mot_rc == 0 else f"FAIL({mot_rc})"
                print(f"[DONE] {spec.run_name} / mot status={mot_status}")

    print("All runs completed.")


if __name__ == "__main__":
    main()
