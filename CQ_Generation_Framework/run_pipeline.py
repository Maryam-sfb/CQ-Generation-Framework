import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "extract_domain_info.py",
    "extract_articles_generate_CQs.py",
    "refinement.py",
    "joint_filtering.py",
]

def run_pipeline():
    base_dir = Path(__file__).resolve().parent

    for script in SCRIPTS:
        script_path = base_dir / script
        if not script_path.exists():
            print(f"[ERROR] Script not found: {script}")
            sys.exit(1)

        print(f"\n[INFO] Running {script} ...")
        result = subprocess.run([sys.executable, str(script_path)])
        if result.returncode != 0:
            print(f"[ERROR] {script} failed with exit code {result.returncode}")
            sys.exit(result.returncode)

    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
