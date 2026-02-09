"""Head-to-head comparison: Psi-Sampler vs GG-RTP-SMC.

Compares at the same total NFE budget:
| Method      | N  | M | eta | Total NFE/step |
|-------------|-----|---|-----|----------------|
| Psi-Sampler | 20 | 1 | -   | 20             |
| GG-RTP-SMC  | 5  | 4 | 1.0 | 20             |
| GG-RTP-SMC  | 4  | 5 | 1.0 | 20             |
| GG-RTP-SMC  | 10 | 4 | 1.0 | 40             |

Evaluates on all 3 tasks: layout-to-image, aesthetic, quantity-aware.

Usage:
    python scripts/compare.py --save_dir ./results/comparison
    python scripts/compare.py --task aesthetic --save_dir ./results/comparison
"""

import os
import sys
import subprocess
import argparse


TASK_CONFIGS = {
    "layout_to_image": {
        "baseline_config": "./config/layout_to_image.yaml",
        "rtp_config": "./config/rtp_layout.yaml",
        "data_path": "./data/layout_to_image.json",
    },
    "aesthetic": {
        "baseline_config": "./config/aesthetic.yaml",
        "rtp_config": "./config/rtp_aesthetic.yaml",
        "data_path": "./data/aesthetic.txt",
    },
    "quantity_aware": {
        "baseline_config": "./config/quantity_aware.yaml",
        "rtp_config": "./config/rtp_quantity.yaml",
        "data_path": "./data/quantity_aware.json",
    },
}

# Comparison configurations: (method_name, config_key, N, M, eta)
COMPARISON_CONFIGS = [
    ("psi_sampler_N20",  "baseline", 20, 1,  0.0),   # Psi-Sampler baseline
    ("rtp_smc_N5_M4",   "rtp",      5,  4,  1.0),   # GG-RTP-SMC (same NFE=20)
    ("rtp_smc_N4_M5",   "rtp",      4,  5,  1.0),   # GG-RTP-SMC (same NFE=20)
    ("rtp_smc_N10_M4",  "rtp",      10, 4,  1.0),   # GG-RTP-SMC (higher NFE=40)
]


def run_experiment(config, data_path, save_dir, method, N, M, eta, tag):
    """Launch a single experiment."""
    cmd = [
        sys.executable, "main.py",
        "--config", config,
        "--data_path", data_path,
        "--save_dir", save_dir,
        "--save_reward",
        "--extra_tag", tag,
        f"num_particles={N}",
    ]

    if method == "rtp":
        cmd += [
            f"method=rtp_smc",
            f"M={M}",
            f"eta={eta}",
        ]

    print(f"[Compare] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Psi-Sampler vs GG-RTP-SMC Comparison")
    parser.add_argument("--task", default="all",
                        choices=["all", "layout_to_image", "aesthetic", "quantity_aware"])
    parser.add_argument("--save_dir", default="./results/comparison")
    args = parser.parse_args()

    tasks = list(TASK_CONFIGS.keys()) if args.task == "all" else [args.task]

    for task_name in tasks:
        task_cfg = TASK_CONFIGS[task_name]
        print(f"\n{'='*60}")
        print(f"[Compare] Task: {task_name}")
        print(f"{'='*60}\n")

        for method_name, config_key, N, M, eta in COMPARISON_CONFIGS:
            config = task_cfg["baseline_config"] if config_key == "baseline" else task_cfg["rtp_config"]
            data_path = task_cfg["data_path"]
            tag = f"compare_{method_name}"

            rc = run_experiment(
                config=config,
                data_path=data_path,
                save_dir=args.save_dir,
                method=config_key,
                N=N, M=M, eta=eta,
                tag=tag
            )

            nfe_per_step = N * M
            status = "OK" if rc == 0 else f"FAILED (code {rc})"
            print(f"  [{status}] {method_name}: N={N}, M={M}, eta={eta}, NFE/step={nfe_per_step}")

    print(f"\n[Compare] All experiments completed. Results in: {args.save_dir}")


if __name__ == "__main__":
    main()
