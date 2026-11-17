import subprocess
import sys
from pathlib import Path


def main() -> None:
    # Run the generator as a checked subprocess so deployment never silently uses stale data.
    data_gen = Path(__file__).parent / "data_gen.py"
    subprocess.run([sys.executable, str(data_gen)], check=True)
    embed = Path(__file__).parent / "embed.py"
    subprocess.run([sys.executable, str(embed)], check=True)
    print("build ok")


if __name__ == "__main__":
    main()
