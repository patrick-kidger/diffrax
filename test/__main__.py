import pathlib
import subprocess
import sys


here = pathlib.Path(__file__).resolve().parent


# Each file is ran separately to avoid out-of-memorying.
running_out = 0
for file in here.iterdir():
    if file.is_file() and file.name.startswith("test"):
        out = subprocess.run(f"pytest {file}", shell=True).returncode
        running_out = max(running_out, out)
sys.exit(running_out)
