import subprocess
import sys
from pathlib import Path


def main():
    app_path = Path(__file__).resolve().parent.parent.parent / "app" / "streamlit_app.py"
    subprocess.call([
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.headless", "true",
        "--server.port", "8510",
        "--server.address", "0.0.0.0",
    ])
