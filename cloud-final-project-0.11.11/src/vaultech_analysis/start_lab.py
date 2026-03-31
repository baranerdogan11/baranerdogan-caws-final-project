import subprocess
import sys


def main():
    subprocess.call([
        sys.executable, "-m", "jupyter", "lab", "notebooks/",
        "--no-browser",
        "--ServerApp.ip=0.0.0.0",
        "--ServerApp.token=",
        "--ServerApp.password=",
        "--ServerApp.disable_check_xsrf=True",
    ])
