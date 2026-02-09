"""Ablation study for GG-RTP-SMC.

Sweeps over:
- M in {1, 2, 4, 8}       (M=1 = Psi-Sampler baseline)
- eta in {0, 0.5, 1.0, 2.0}  (eta=0 = no gradient guidance in candidates)
- N x M budget: fix total NFE, vary N vs M ratio

Usage:
    python scripts/ablation.py --config ./config/rtp_layout.yaml \
        --data_path ./data/layout_to_image.json \
        --save_dir ./results/ablation
"""

import os
import sys
import subprocess
import argparse
import itertools


def run_experiment(config, data_path, save_dir, M, eta, num_particles, extra_tag=None):
    """Launch a single experiment with given hyperparameters."""
    tag = f"M{M}_eta{eta}_N{num_particles}"
    if extra_tag:
        tag = f"{extra_tag}_{tag}"

    cmd = [
        sys.executable, "main.py",
        "--config", config,
        "--data_path", data_path,
        "--save_dir", save_dir,
        "--save_reward",
        "--extra_tag", tag,
        f"method=rtp_smc",
        f"M={M}",
        f"eta={eta}",
        f"num_particles={num_particles}",
    ]

    print(f"[Ablation] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="GG-RTP-SMC Ablation Study")
    parser.add_argument("--config", default="./config/rtp_layout.yaml")
    parser.add_argument("--data_path", default="./data/layout_to_image.json")
    parser.add_argument("--save_dir", default="./results/ablation")
    parser.add_argument("--sweep", default="M_eta",
                        choices=["M_eta", "budget"],
                        help="Which sweep to run: M_eta or budget")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.sweep == "M_eta":
        # Sweep M and eta independently
        M_values = [1, 2, 4, 8]
        eta_values = [0.0, 0.5, 1.0, 2.0]
        num_particles = 5  # fixed

        for M, eta in itertools.product(M_values, eta_values):
            # Skip redundant: M=1 with eta>0 is same as M=1 eta=0
            # (gradient shift has no effect when there's only 1 candidate
            #  and selection is trivial)
            if M == 1 and eta > 0:
                continue
            rc = run_experiment(args.config, args.data_path, args.save_dir,
                                M=M, eta=eta, num_particles=num_particles,
                                extra_tag="sweep_M_eta")
            if rc != 0:
                print(f"[Ablation] WARNING: Experiment M={M}, eta={eta} failed with code {rc}")

    elif args.sweep == "budget":
        # Fixed total NFE per step = N * M = 20
        total_nfe = 20
        configs = [
            (20, 1),   # Psi-Sampler baseline
            (10, 2),
            (5, 4),
            (4, 5),
            (2, 10),
        ]
        eta = 1.0  # fixed

        for num_particles, M in configs:
            assert num_particles * M == total_nfe, f"N*M must equal {total_nfe}"
            rc = run_experiment(args.config, args.data_path, args.save_dir,
                                M=M, eta=eta, num_particles=num_particles,
                                extra_tag="sweep_budget")
            if rc != 0:
                print(f"[Ablation] WARNING: Experiment N={num_particles}, M={M} failed with code {rc}")

    print("[Ablation] All experiments completed.")


if __name__ == "__main__":
    main()
