"""
Batch runner for the revision campaign: loops over models (and optionally
seeds) on one dataset, reusing the same leakage-safe split within a
(dataset, seed) pair so every model in that pass is compared on identical
train/val/test data. Failures on one model are logged and do not abort the run.

Example:
    python scripts/run_campaign.py --dataset CVC-ClinicDB --seeds 0 --epochs 100
"""
import argparse
import csv
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import ALL_MODELS, build_data, run_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["CVC-ClinicDB", "Kvasir-SEG", "PolypGen"])
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--group_strategy", default="phash", choices=["phash", "filename_prefix", "none"])
    ap.add_argument("--base_dir", default="results/runs")
    ap.add_argument("--split_seed", type=int, default=None,
                    help="Fixed partition seed shared by all training seeds (default: re-split per seed)")
    ap.add_argument("--device", default=None, help="Force a device (e.g. 'cpu') to avoid contending with a concurrent GPU job")
    args = ap.parse_args()

    summary_path = Path(args.base_dir) / "campaign_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_path.exists()

    # Resume support: skip (model, dataset, seed) triples already logged as ok.
    done = set()
    if summary_path.exists():
        with open(summary_path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("status") == "ok":
                    done.add((r["model"], r["dataset"], int(r["seed"])))

    for seed in args.seeds:
        gs = None if args.group_strategy == "none" else args.group_strategy
        ss = seed if args.split_seed is None else args.split_seed
        ds = build_data(args.dataset, args.batch_size, ss, gs)
        print(f"\n=== dataset={args.dataset} seed={seed} ===")
        print(ds.report.summary())
        split_report_path = Path(args.base_dir) / "split_reports"
        split_report_path.mkdir(parents=True, exist_ok=True)
        import json
        with open(split_report_path / f"{args.dataset}_split{ss}_trainseed{seed}.json", "w") as f:
            json.dump(ds.report.__dict__, f, indent=2, default=str)

        for model_name in args.models:
            if (model_name, args.dataset, seed) in done:
                print(f"\n--- SKIP {model_name} | {args.dataset} | seed={seed} (already ok in summary) ---")
                continue
            print(f"\n--- {model_name} | {args.dataset} | seed={seed} ---")
            try:
                results = run_one(
                    model_name, args.dataset, seed=seed, epochs=args.epochs,
                    batch_size=args.batch_size, base_dir=args.base_dir, ds=ds,
                    device=args.device, split_seed=ss,
                )
                row = {
                    "model": model_name, "dataset": args.dataset, "seed": seed,
                    "status": "ok",
                    "dice": results["avg_dice_score"], "iou": results["avg_iou"],
                    "precision": results["precision"], "recall": results["recall"],
                    "f1": results["f1"], "precision_micro": results["precision_micro"],
                    "recall_micro": results["recall_micro"], "f1_micro": results["f1_micro"],
                    "params": results["total_params"], "gflops": results["gflops"],
                    "fps": results["fps"], "error": "",
                }
            except Exception as e:
                traceback.print_exc()
                row = {
                    "model": model_name, "dataset": args.dataset, "seed": seed,
                    "status": "FAILED", "dice": "", "iou": "", "precision": "", "recall": "",
                    "f1": "", "precision_micro": "", "recall_micro": "", "f1_micro": "",
                    "params": "", "gflops": "", "fps": "", "error": repr(e),
                }
            with open(summary_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerow(row)
            print(f"Logged to {summary_path}: {row['status']}")


if __name__ == "__main__":
    main()
