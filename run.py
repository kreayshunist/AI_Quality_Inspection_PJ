import argparse, sys, subprocess, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
def sh(cmd, cwd=None):
    print("[cmd]", " ".join(map(str, cmd)), f"(cwd={cwd})")
    subprocess.run(cmd, check=True, cwd=cwd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--epochs", type=int, default=None)  
    ap.add_argument("--use-cluster", type=int, default=0)
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=None,
                help="Default: F1AdaptiveThreshold")
    ap.add_argument("--masked", type=str, default="zero_out", help="Usage: --masked zero_out | crop")


    args = ap.parse_args()

    root = Path.cwd()
    (root / "output").mkdir(parents=True, exist_ok=True)

    # Run run_sam2.py
    sh([
        sys.executable, "src/run_sam2.py",
        "--device", args.device,
        "--use-cluster", str(args.use_cluster),
        "--num-gpus", str(args.num_gpus),
        *( ["--epochs", str(args.epochs)] if args.epochs is not None else [] ),
    ])

    print("\nSAM2 done | Check: 'output/'\n")

    os.chdir(PROJECT_ROOT)
    print(f"Returned to project root: {os.getcwd()}")


    # Run run_patchcore.py
    cmd = [
        sys.executable, "src/run_patchcore.py",
        "--masked", str(args.masked),
    ]
    if args.threshold is not None:
        cmd += ["--threshold", str(args.threshold)]
    sh(cmd, cwd=PROJECT_ROOT)

    print("\nPatchCore done | Check: 'output/'\n")

if __name__ == "__main__":
    main()
